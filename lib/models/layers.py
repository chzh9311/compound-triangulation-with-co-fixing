import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.functions import heatmap_normalize, heatmap_std, calc_avg_direction, sed_selection, normalize, vec_selection

BN_MOMENTUM = 0.1


class SoftArgmax(nn.Module):
    def __init__(self, h, w, beta):
        """
        heatmap: ... x h x w
        grids: pre-calculated meshgrids.
        equation: output = \sum_{x}\frac{exp{\beta h(x)}}{\sum_{x}exp{\beta h(x)}}x
        """
        super(SoftArgmax, self).__init__()
        self.gridy, self.gridx = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        # self.gridy, self.gridx = gridy.to(device).float(), gridx.to(device).float()
        self.beta = beta
        self.hm_shape = (h, w)

    
    def forward(self, heatmap):
        heatmap = heatmap.view(*heatmap.shape[:-2], -1)

        soft_hms = F.softmax(heatmap * self.beta, dim=-1).view(heatmap.shape[:-1] + self.hm_shape)

        norm = torch.sum(soft_hms, dim=(-1, -2))
        kp_y = torch.sum(soft_hms * self.gridy.to(heatmap.device), dim=(-1, -2)) / norm
        kp_x = torch.sum(soft_hms * self.gridx.to(heatmap.device), dim=(-1, -2)) / norm

        return torch.stack((kp_x, kp_y), dim=-1)


class RefineLayers(nn.Module):

    def __init__(self, n_features, p=0.5):
        super(RefineLayers, self).__init__()
        self.residual = nn.Sequential(
            nn.Linear(n_features, n_features, bias=True),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(n_features, n_features, bias=True),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Dropout(p=p)
        )

    def forward(self, x):
        out = self.residual(x)
        out = out + x

        return out


class GlobalAveragePoolingHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x) # 256 x 12 x 12

        batch_size, n_channels = x.shape[:2] # 256 x 3 x 3
        x = x.view((batch_size, n_channels, -1)) # 256 x 9
        x = x.mean(dim=-1) # 256 x 1

        out = self.head(x) # 17

        return out


class rdsvd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        # s, u = th.symeig(M, eigenvectors=True, upper=True)  # s in a ascending sequence.
        ut, s, u = torch.svd(M)  # s in a descending sequence.
        # print('0-eigenvalue # {}'.format((s <= 1e-5).sum()))
        s = torch.clamp(s, min=1e-10)  # 1e-5
        ctx.save_for_backward(M, u, s)
        return u, s

    @staticmethod
    def backward(ctx, dL_du, dL_ds):
        M, u, s = ctx.saved_tensors
        # I = torch.eye(s.shape[0])
        # K_t = 1.0 / (s[None] - s[..., None] + I) - I
        K_t = geometric_approximation(s).transpose(-1, -2)
        u_t = u.transpose(-1, -2)
        diag = torch.zeros_like(dL_du, device=s.device)
        diag[..., list(range(s.shape[-1])), list(range(s.shape[-1]))] = dL_ds
        dL_dM = u @ (K_t * (u_t @ dL_du) + diag) @ u_t
        return dL_dM

    
def geometric_approximation(s, K=9):
    # batch_size x n_limbs x 3
    dtype = s.dtype
    I = torch.eye(s.shape[-1], device=s.device).type(dtype).view(1, 1, s.shape[-1], s.shape[-1])
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = torch.where(p < 1., p, 1. / p) # If p < 1, yield p, else yield 1/p
    a1_t = s.unsqueeze(-2).repeat(1, 1, s.shape[-1], 1)
    a1 = a1_t.transpose(-1, -2)
    a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t) # anti-symmetric. The foremost 1 / lambda coefficient.
    a1 *= torch.ones_like(I, device=s.device).type(dtype) - I # eliminate the diagon
    p_app = torch.ones_like(p, device=s.device)
    p_hat = torch.ones_like(p, device=s.device)
    for i in range(K): # Taylor expansion level
        p_hat = p_hat * p
        p_app += p_hat
    a1 = a1 * p_app
    return a1


class CoFixing(nn.Module):
    def __init__(self, heatmap_size, n_joints, limb_pairs, alpha, beta):
        super(CoFixing, self).__init__()
        self.heatmap_size = heatmap_size
        self.n_joints = n_joints
        self.limb_pairs = limb_pairs
        self.n_limbs = limb_pairs.shape[0]
        h, w = heatmap_size
        coords = torch.stack(torch.meshgrid(torch.arange(2*h-1), torch.arange(2*w-1)), dim=0).transpose(1, 2)
        centre = torch.tensor([h, w]).view(2, 1, 1)
        ker = (centre - coords).float()#.to(device).float()
        dist = torch.norm(ker, dim=0, keepdim=True)
        dist[0, h, w] = 1
        self.soft_argmax = SoftArgmax(h, w, beta)
        self.kernel1 = (ker / dist ** (1+alpha)).view(1, 1, 2, 2*h-1, 2*w-1)
        self.kernel2 = (ker / dist ** (1+alpha)).view(1, 2, 2*h-1, 2*w-1)
        # self.kernel1 /= (h * w - 1)
    
    def forward(self, heatmaps, fields, Ps, Rs, Cs, image_shape, confs, vth1, vth2, pth1, pth2):
        """
        Fuse joint heatmaps and limb fields in 2D level
        heatmap: batch_size x n_views x n_joints x h x w
        fields: batch_size x n_views x n_limbs x 3 x h x w
        Rs: batch_size x n_views x 3 x 3
        image_shape: tuple (h, w)
        confs: batch_size x n_views x (n_limbs + n_joints)
        """
        bs, nv, nb, _, h, w = fields.shape
        self.kernel1 = self.kernel1.to(heatmaps.device)
        self.kernel2 = self.kernel2.to(heatmaps.device)
        heatmaps = heatmaps - torch.mean(heatmaps, dim=(3, 4), keepdim=True)
        # fields = fields.view(bs, nv, self.n_limbs, 3, h, w)
        std_hm = heatmap_std(heatmaps, di=False)
        std_df = heatmap_std(fields, di=True)

        Hs = heatmaps[:, :, self.limb_pairs[:, 1], :, :] - heatmaps[:, :, self.limb_pairs[:, 0], :, :]
        convs = F.conv3d(self.kernel1, Hs.view(bs*nv*self.n_limbs, 1, 1, h, w)[..., range(h-1, -1, -1), :][..., :, range(w-1, -1, -1)], stride=1, padding=0)
        # conv_y = F.conv2d(self.kernel[:, 1].unsqueeze(0), Hs[..., range(h-1, -1, -1), :][..., :, range(w-1, -1, -1)], stride=1, padding=0)
        # convs = torch.stack((conv_x, conv_y), dim=2)
        convs = convs.view(bs, nv, self.n_limbs, 2, h, w)
        coef = torch.sum(convs * fields[..., :2, :, :], dim=3, keepdim=True)
        coef = F.relu(coef)
        fields_new = fields * coef

        ## Determine whether to apply field change using direction variation.
        lvecs = torch.sum(torch.norm(fields, dim=3, keepdim=True) * fields, dim=(4, 5)).unsqueeze(-1)
        lvecs = normalize(lvecs, dim=3).squeeze(-1)
        lpts = self.soft_argmax(torch.norm(fields, dim=3))
        lpts = torch.stack((lpts[:, :, :, 0] * image_shape[1] / w,
                            lpts[:, :, :, 1] * image_shape[0] / h), dim=3)
        lpts_new = self.soft_argmax(torch.norm(fields_new, dim=3))
        lpts_new = torch.stack((lpts_new[:, :, :, 0] * image_shape[1] / w,
                                lpts_new[:, :, :, 1] * image_shape[0] / h), dim=3)
        lvecs_new = torch.sum(torch.norm(fields_new, dim=3, keepdim=True) * fields_new, dim=(4, 5)).unsqueeze(-1)
        lvecs_new = normalize(lvecs_new, dim=3).squeeze(-1)
        delta = vec_selection(lpts, lvecs, lpts_new, lvecs_new, confs[:, :, :nb], Rs, Cs, nv, vth1, vth2)
        delta[:] = 0
        comb_v = delta * lvecs_new + (1-delta) * lvecs
        comb_p = delta * lpts_new + (1-delta) * lpts
        delta = delta.unsqueeze(-1).unsqueeze(-1)
        fields = heatmap_normalize(fields * (1 - delta) + fields_new * delta, std_df, di=True)

        ## Calculate fixed heatmaps
        # fields = heatmap_normalize(fields_new, std_df, di=True)
        fds2jts = F.conv2d(self.kernel2, fields.view(bs*nv*self.n_limbs, 3, h, w)[:, :2])[..., range(h-1, -1, -1), :][..., :, range(w-1, -1, -1)]
        # prox_ = F.softmax(-fds2jts.view(bs, self.n_limbs, h*w), dim=-1).view(bs, self.n_limbs, h, w)
        # dist_ = F.softmax(fds2jts.view(bs, self.n_limbs, h*w), dim=-1).view(bs, self.n_limbs, h, w)
        prox_ = F.relu(-fds2jts).view(bs, nv, self.n_limbs, h, w)
        dist_ = F.relu(fds2jts).view(bs, nv, self.n_limbs, h, w)
        hms_new = heatmaps.clone()
        hms_new[:, :, self.limb_pairs[:, 0], :, :] = hms_new[:, :, self.limb_pairs[:, 0], :, :] * prox_
        hms_new[:, :, self.limb_pairs[:, 1], :, :] = hms_new[:, :, self.limb_pairs[:, 1], :, :] * dist_
        hms_new = heatmap_normalize(hms_new, std_hm)

        ## Determine whether to apply fixed heatmaps
        ### calculate points
        kps = self.soft_argmax(heatmaps)
        kps = torch.stack((kps[:, :, :, 0] * image_shape[1] / w,
                           kps[:, :, :, 1] * image_shape[0] / h), dim=3).unsqueeze(-1)
        kps_new = self.soft_argmax(hms_new)
        kps_new = torch.stack((kps_new[:, :, :, 0] * image_shape[1] / w,
                               kps_new[:, :, :, 1] * image_shape[0] / h), dim=3).unsqueeze(-1)
        ### replace possible nans
        kps_new[torch.isnan(kps_new)] = kps[torch.isnan(kps_new)]
        kps_combined = sed_selection(Ps, Cs, kps, kps_new, nv, confs[:, :, nb:], pth1, pth2)
        
        return kps_combined, comb_v, comb_p, fields_new, hms_new


# class ModularFusion(nn.Module):
#     def __init__(self, heatmap_size, n_joints, limb_pairs, alpha, beta):
#         # enable for E2E training.
#         super(ModularFusion, self).__init__()
#         self.heatmap_size = heatmap_size
#         self.n_joints = n_joints
#         self.limb_pairs = limb_pairs
#         self.n_limbs = limb_pairs.shape[0]
#         h, w = heatmap_size
#         coords = torch.stack(torch.meshgrid(torch.arange(2*h-1), torch.arange(2*w-1), indexing='xy'), dim=0)
#         centre = torch.tensor([h, w]).view(2, 1, 1)
#         ker = (centre - coords).float()#.to(device).float()
#         dist = torch.norm(ker, dim=0, keepdim=True)
#         dist[0, h, w] = 1
#         self.softmax_beta = beta
#         self.kernel1 = (ker / dist ** (1+alpha)).view(1, 1, 2, 2*h-1, 2*w-1)
#         self.kernel2 = (ker / dist ** (1+alpha)).view(1, 2, 2*h-1, 2*w-1)