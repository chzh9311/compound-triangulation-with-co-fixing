import torch
from torch import nn
torch.backends.cuda.preferred_linalg_library("cusolver")

from easydict import EasyDict as edict

from lib.models.field_pose_net import get_FPNet
from lib.models.layers import rdsvd, CoFixing, SoftArgmax
from lib.utils.DictTree import create_human_tree
from lib.utils.functions import *
from lib.models.structural_triangulation import stri_from_opt


class MultiViewFPNet(nn.Module):
    """
    Multi-view field pose net.
    """

    def __init__(self, cfg, is_train):
        super(MultiViewFPNet, self).__init__()
        self.is_train = is_train
        self.backbone = get_FPNet(cfg, is_train)
        self.soft_argmax = SoftArgmax(*cfg.MODEL.EXTRA.HEATMAP_SIZE, cfg.MODEL.SOFTMAX_BETA)

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.num_limbs = cfg.MODEL.NUM_LIMBS
        self.softmax_beta = cfg.MODEL.SOFTMAX_BETA
        self.use_lof = cfg.MODEL.USE_LOF
        self.backbone_out_label = cfg.MODEL.BACKBONE_OUTPUT
        self.backbone_type = cfg.MODEL.BACKBONE
        self.model_out_label = cfg.MODEL.MODEL_OUTPUT
        self.is_train = is_train
        self.field_dim = cfg.MODEL.NUM_DIMS
        self.use_gt = False
        self.lam = cfg.MODEL.LAMBDA
        self.svd_layer = rdsvd.apply
        self.svd_eps = cfg.TRAIN.SVD_EPS
        self.fusion_layer = CoFixing(heatmap_size=cfg.MODEL.EXTRA.HEATMAP_SIZE, n_joints=cfg.MODEL.NUM_JOINTS,
                                     limb_pairs=create_human_tree(cfg.DATASET.NAME).limb_pairs,
                                     alpha=cfg.MODEL.CO_FIXING.FIX_ALPHA, beta=cfg.MODEL.SOFTMAX_BETA)
        self.fix_ths = [cfg.MODEL.CO_FIXING.VEC_IMPROVE_TH, cfg.MODEL.CO_FIXING.VEC_FIX_UB, 
                        cfg.MODEL.CO_FIXING.PTS_IMPROVE_TH, cfg.MODEL.CO_FIXING.PTS_FIX_UB]

    # @profile
    def forward(self, htree, images, projections, intrinsics=None, rotation=None, cam_ctr=None, gt_label=None, fix_heatmap=False, **kwargs):
        """
        images: batch_size x n_views x n_channels x h x w.
        projections: batch_size x n_views x 3 x 4.
        htree: <DictTree> The human tree structure.
        """
        # change shapes
        batch_size, n_views = images.shape[:2]
        images_shape = images.shape
        images = images.view(batch_size * n_views, *images.shape[2:])

        backbone_out = self.backbone(images)
        out_values = edict()
        model_out = []
        if intrinsics is not None:
            out_values.intrinsics = intrinsics
        if rotation is not None:
            out_values.rotation = rotation
        if cam_ctr is not None:
            out_values.cam_ctr = cam_ctr
        if gt_label is not None:
            out_values.di_vectors, out_values.line_pos = gt_label
            # out_values.di_vectors = gt_label
            self.use_gt = True
        # r = torch.zeros((batch_size,), device=images.device)
        for i, k in enumerate(self.backbone_out_label):
            out_values[k] = backbone_out[i] if len(self.backbone_out_label) > 1 else backbone_out
        out_values.heatmap = out_values.heatmap.view(batch_size, n_views, -1, *out_values.heatmap.shape[-2:])
        h, w = out_values.heatmap.shape[-2:]
        if "lof" in out_values:
            out_values.lof = out_values.lof.view(batch_size, n_views, -1, *out_values.lof.shape[-2:])
        if "confidences" in out_values:
            out_values.confidences =  out_values.confidences.view(batch_size, n_views, -1)
        # for joint training.
        device = out_values.heatmap.device
        if "data_label" in kwargs:
            n_joints = kwargs["num_joints"]
            n_limbs = kwargs["num_limbs"]
            heatmap = torch.zeros(batch_size, n_views, n_joints, h, w, device=device)
            lof = torch.zeros(batch_size, n_views, n_limbs*3, h, w, device=device)
            confidences = torch.zeros(batch_size, n_views, n_joints + n_limbs, device=device)

            mask = kwargs['data_label'].flatten() == 0
            heatmap[mask] = out_values.heatmap[mask, :, :n_joints]
            lof[mask] = out_values.lof[mask, :, :n_limbs*3]
            confidences[mask] = out_values.confidences[mask, :, :n_joints + n_limbs]

            heatmap[~mask] = out_values.heatmap[~mask, :, n_joints:]
            lof[~mask] = out_values.lof[~mask, :, n_limbs*3:]
            confidences[~mask] = out_values.confidences[~mask, :, n_joints + n_limbs:]
            # Caution: only valid when n_joints1 == n_joints2
            self.num_joints = n_joints
            self.num_limbs = n_limbs
            out_values.lof = lof
            out_values.confidences = confidences
            out_values.heatmap = heatmap
        Nj, Nl = self.num_joints, self.num_limbs

        if "confidences" in self.backbone_out_label:
            out_values.confidences = out_values.confidences + 0.0001 # avoid singularity
        else:
            if "lof" in out_values:
                out_values.confidences = torch.ones(batch_size, n_views, Nj + Nl, device=device) / n_views
            else:
                out_values.confidences = torch.ones(batch_size, n_views, Nj, device=device) / n_views

        kps_in_hm = self.soft_argmax(out_values.heatmap) # b, nv, njoint, 2
        # kps_in_hm = kps_in_hm[:, :, :, [1, 0]]

        kps = torch.stack((kps_in_hm[:, :, :, 1] * images_shape[3] / w,
                           kps_in_hm[:, :, :, 0] * images_shape[4] / h), dim=3).unsqueeze(-1)

        out_values.keypoints2d = kps
        if not self.use_gt:
            if fix_heatmap:
                kps_combined, di_combined, limb_kps_combined, dm_fixed, hm_fixed = self.fusion_layer(
                    out_values.heatmap, out_values.lof, projections, rotation, cam_ctr, images_shape[3:], out_values.confidences,
                    *self.fix_ths)
                out_values.di_combined = di_combined
            # limb_pairs = np.array(htree.limb_pairs)

            # kps = torch.stack((kps_in_hm[:, :, :, 0] * images_shape[4] / w,
            #                 kps_in_hm[:, :, :, 1] * images_shape[3] / h), dim=3).unsqueeze(-1)

        if "bone_lengths" not in kwargs:
            bls = None
            sca_steps = 3
        else:
            bls = kwargs['bone_lengths']
            sca_steps = kwargs['sca_steps']
        if self.use_lof:
            limb_pairs = np.array(htree.limb_pairs)
            di_maps = out_values.lof.view(batch_size, n_views, Nl, self.field_dim, h, w)
            di = calc_avg_direction(kps_in_hm, di_maps, limb_pairs, -1)
            out_values.di_vectors = di
            norm_maps = torch.norm(di_maps, dim=3)
            pos_in_hm = self.soft_argmax(norm_maps)
            limb_kps = torch.stack((pos_in_hm[:, :, :, 1] * images_shape[3] / h,
                                pos_in_hm[:, :, :, 0] * images_shape[4] / w), dim=3)
            kps_3d = optimize_wrt_params(kps, projections, htree, self.use_lof, confidences=out_values.confidences,
                                        intrinsics=intrinsics, rotation=rotation, di_vectors=di, line_pos=limb_kps,
                                        bone_lengths=bls, sca_steps=sca_steps)
        else:
            kps_3d = optimize_wrt_params(kps, projections, htree, self.use_lof, confidences=out_values.confidences,
                                        intrinsics=intrinsics, rotation=rotation,
                                        bone_lengths=bls, sca_steps=sca_steps)

        model_out = [kps_3d]
        # if "keypoints3d_tri" in self.model_out_label:
        #     out_values.keypoints3d_tri = kps_3d[1]
            # if "lines_tri" in self.model_out_label:
            #     out_values.lines_tri = kps_3d[2]

        for k in self.model_out_label:
            if k == "lof":
                model_out.append(out_values[k].view(batch_size, n_views, self.num_limbs, self.field_dim, *out_values.lof.shape[-2:]))
            elif k == "keypoints3d":
                pass
            elif k == "keypoints3d_tri":
                kps_tri = optimize_wrt_params(kps, projections, htree, False, confidences=out_values.confidences,
                                              intrinsics=intrinsics, rotation=rotation,
                                              bone_lengths=bls, sca_steps=sca_steps)
                model_out.append(kps_tri)

            else:
                model_out.append(out_values[k])
        if fix_heatmap:
            kps_3d_combined = optimize_wrt_params(kps_combined, projections, htree, self.use_lof,
                                                    confidences=out_values.confidences, intrinsics=intrinsics, rotation=rotation,
                                                    di_vectors=di_combined, line_pos=limb_kps_combined,
                                                    bone_lengths=bls, sca_steps=sca_steps)
            model_out += [kps_combined, kps_3d_combined, dm_fixed, hm_fixed]
        return model_out

def get_MVFPNet(config, is_train=False):
    model = MultiViewFPNet(config, is_train)


# @profile
def optimize_wrt_params(kps_2d, projections, htree, use_lof, **kwargs):
    """
    kps_2d: batch_size x n_views x n_joints x 2 x 1
    projections: batch_size x n_views x 3 x 4
    mus: batch_size x n_views x n_bones
    returns: batch_size x n_joints x 3
    """

    # First concatenate inner points with key points.
    batch_size, n_views, Nj = kps_2d.shape[:3]
    Nl = htree.limb_pairs.shape[0]
    device = projections.device
    if use_lof:
        # norm_mus = torch.zeros(batch_size, n_views, Nj, device=device)
        # norm_mus[:, :, htree.root["index"]] = 1
        # for k in range(5):
        #     for i, (px, dt) in enumerate(htree.limb_pairs):
        #         norm_mus[:, :, dt] = norm_mus[:, :, px] * kwargs["mus"][:, :, i]
        ws = kwargs["confidences"].view(batch_size, n_views, -1, 1, 1)
        ws_lms = ws[:, :, :Nl, :, :]
        ws_kps = ws[:, :, Nl:, :, :]

        KRs = projections[:, :, :, :3].unsqueeze(2)
        homo_kps_2d = torch.cat((kps_2d, torch.ones(batch_size, n_views, Nj, 1, 1, device=device)), dim=-2)
        K = kwargs["intrinsics"].unsqueeze(2)
        K_inv = torch.inverse(K)
        Cs = - torch.linalg.solve(projections[:, :, :, :3], projections[:, :, :, 3:4]).view(batch_size, n_views, -1, 1)
        C = torch.cat([Cs for i in range(Nj)], dim=2)

        # Core codes
        # Mahas = generate_Mahalanobis_mat((K_inv @ homo_kps_2d - unit_z_vec).squeeze(-1), (K_inv @ homo_kps_2d).squeeze(-1))
        Mahas = torch.eye(3, device=device).view(1, 1, 1, 3, 3) * torch.ones(1, n_views, Nj, 1, 1, device=device)
        # Mahas = generate_Mahalanobis_mat((homo_kps_2d - K @ unit_z_vec).squeeze(-1), homo_kps_2d.squeeze(-1))
        Mahas = K_inv.transpose(-1, -2) @ Mahas @ K_inv
        # Mahas = torch.concat([K_inv.transpose(-1, -2) @ K_inv for i in range(Nj)], dim=2)
        # Mahas = torch.stack([torch.eye(3, device=device, dtype=torch.float32) for i in range(Nj)], dim=0).view(1, 1, Nj, 3, 3)
        D1 = block_diag_batch(ws_kps * KRs.transpose(-1, -2) @ Mahas @ KRs)

        if "di_vectors" in kwargs and "line_pos" in kwargs:
            D2 = ws_kps * KRs.transpose(-1, -2) @ Mahas @ homo_kps_2d @ homo_kps_2d.transpose(-1, -2) @ Mahas @ KRs / \
                 (homo_kps_2d.transpose(-1, -2) @ Mahas @ homo_kps_2d)
            D2 = block_diag_batch(D2)
            di_vectors = kwargs["di_vectors"]
            line_pos = kwargs["line_pos"]
            homo_kps_2d = torch.cat((line_pos, torch.ones(*line_pos.shape[:3], 1, device=device)), dim=3)
            pos_vec_3d = torch.linalg.solve(K.repeat(1, 1, Nl, 1, 1), homo_kps_2d.unsqueeze(-1))
            D3 = line_set_triangulation(pos_vec_3d, di_vectors, htree.limb_pairs, kwargs["rotation"], ws_lms, Nj, True)

            # Compound Triangulation
            D = torch.sum(D1 - D2 + D3, dim=1)
            DC = torch.sum((D1 - D2 + D3) @ C, dim=1)

            # point triangulation
                # if "bone_lengths" in kwargs and kwargs['bone_lengths'] is not None:
                #     X = stri_from_opt(torch.sum(D1 - D2, dim=1), torch.sum((D1 - D2) @ C, dim=1),
                #                            torch.tensor(htree.conv_B2J, device=device).unsqueeze(0).float(),
                #                            kwargs["bone_lengths"], batch_size, Nj, htree.root["index"], kwargs["sca_steps"], device)
                # else:
                    # X = solve(torch.sum(D1 - D2, dim=1), torch.sum((D1 - D2) @ C, dim=1)).view(batch_size, Nj, 3)

        elif "di_vectors" in kwargs and "line_pos" not in kwargs:
            # di_vectors: bs x nv x Nl x 3 x 1
            # Ks: bs x nv x 3 x 3
            # Mahas: bs x nv x Nj x 3 x 3
            # KRs: bs x nv x 1 x 3 x 3
            D2 = ws_kps * KRs.transpose(-1, -2) @ Mahas @ homo_kps_2d @ homo_kps_2d.transpose(-1, -2) @ Mahas @ KRs / \
                 (homo_kps_2d.transpose(-1, -2) @ Mahas @ homo_kps_2d)
            D2 = block_diag_batch(D2)
            DC = torch.sum((D1 - D2) @ C, dim=1)
            Madjs = []
            KRs = KRs.squeeze(2)
            for l in range(Nl):
                Adjs = torch.zeros((1, 1, 3, 3 * Nj), device=device)
                px, dt = htree.limb_pairs[l]
                Adjs[:, :, :, 3*px:3*(px+1)] = torch.eye(3)
                Adjs[:, :, :, 3*dt:3*(dt+1)] = -torch.eye(3)
                mij = Adjs.transpose(-1, -2) @ KRs.transpose(-1, -2) @ Mahas[:, :, px, ...] @ KRs @ Adjs
                d_proj = (K @ kwargs["di_vectors"])[:, :, l, ...]
                mij = mij - Adjs.transpose(-1, -2) @ KRs.transpose(-1, -2) @ Mahas[:, :, px, ...] @ d_proj @ \
                      d_proj.transpose(-1, -2) @ Mahas[:, :, px, ...] @ KRs @ Adjs / (d_proj.transpose(-1, -2) @ Mahas[:, :, px, ...] @ d_proj)

                mji = Adjs.transpose(-1, -2) @ KRs.transpose(-1, -2) @ Mahas[:, :, dt, ...] @ KRs @ Adjs
                mji = mji - Adjs.transpose(-1, -2) @ KRs.transpose(-1, -2) @ Mahas[:, :, dt, ...] @ d_proj @ \
                      d_proj.transpose(-1, -2) @ Mahas[:, :, dt, ...] @ KRs @ Adjs / (d_proj.transpose(-1, -2) @ Mahas[:, :, dt, ...] @ d_proj)
                Madjs.append(ws_lms[:, :, l] * (ws_kps[:, :, px] * mij + ws_kps[:, :, dt] * mji))
            D3 = sum(Madjs)
            D = torch.sum(D1 - D2 + D3, dim=1)

        else:
            # Fuse nothing.
            D2 = ws * KRs.transpose(-1, -2) @ Mahas @ homo_kps_2d @ homo_kps_2d.transpose(-1, -2) @ Mahas @ KRs / \
                 (homo_kps_2d.transpose(-1, -2) @ Mahas @ homo_kps_2d)
            D2 = block_diag_batch(D2)
            D = torch.sum(D1 - D2, dim=1)
            # D = KRs - homo_kps_2d @ homo_kps_2d.transpose(-1, -2) @ Mahas @ KRs / (homo_kps_2d.transpose(-1, -2) @ Mahas @ homo_kps_2d)

            ## Reproduce linear triangulation
            # p3 = projections[:, :, 2:3, :3].unsqueeze(2)
            # D = KRs - homo_kps_2d @ p3
            # D = D.transpose(-1, -2) @ Mahas @ D
            # D = block_diag_batch(D)
            DC = torch.sum(D @ C, dim=1)

        if "bone_lengths" in kwargs and kwargs['bone_lengths'] is not None:
            X = stri_from_opt(D, DC, torch.tensor(htree.conv_B2J, device=device).unsqueeze(0).float(),
                              kwargs["bone_lengths"], batch_size, Nj, htree.root["index"], kwargs["sca_steps"], device)
        else:
            X = torch.linalg.solve(D, DC)

    else:
        fill = torch.zeros(kps_2d.shape[:3] + (2, 2), device=device)
        fill[:, :, :, [0, 1], [0, 1]] = 1
        M_inner = torch.cat((fill, -kps_2d), dim=-1)
        M_half = M_inner @ projections[:, :, :, :3].unsqueeze(2)
        Ms = 2 * M_half.transpose(-1, -2) @ M_half
        m = -2 * M_half.transpose(-1, -2) @ M_inner @ projections[:, :, :, 3:4].unsqueeze(2)

        ws = kwargs["confidences"].view(batch_size, n_views, -1, 1, 1)
        # ws /= torch.sum(ws, dim=1, keepdim=True)
        # ws += 0.000001
        ws_kps = ws[:, :, :Nj, :, :]
        ws_lms = ws[:, :, Nj:, :, :]
        Ms = ws_kps * Ms
        m = ws_kps * m
        Ms = torch.sum(Ms, dim=1)
        D = block_diag_batch(Ms)
        m = torch.sum(m, dim=1).view(batch_size, -1, 1)

        if "bone_lengths" not in kwargs or kwargs["bone_lengths"] is None:
            X = torch.linalg.solve(D, m)
        else:
            X = stri_from_opt(D, m, torch.tensor(htree.conv_B2J, device=device).unsqueeze(0).float(),
                              kwargs["bone_lengths"], batch_size, Nj, htree.root["index"], kwargs["sca_steps"], device)
        
    kps_3d = X.view(batch_size, Nj, 3)
    return kps_3d


def line_set_triangulation(pts, nvs, limb_pairs, Rs, w, n_joints, homo=True):
    """
    Calculate the distance between limbs and estimated limb vectors.
    pts: batch_size x n_views x n_limbs x 3. The position indicator.
    nvs: batch_size x n_views x n_limbs x 3. The limb direction vector.
    if the last dim == 2, it is on the image plane.
    Rs: batch_size x n_views x 3 x 3
    limb_pairs: n_limbs x 2
    w: batch_size x n_views x n_limbs: the confidences.
    """
    device = pts.device
    batch_size, n_views, n_limbs = pts.shape[:3]
    I = torch.eye(3, device=device).view(1, 1, 1, 3, 3)
    if not nvs.shape[-1] == 1:
        nvs = nvs.unsqueeze(-1)
    if not pts.shape[-1] == 1:
        pts = pts.unsqueeze(-1)
    if nvs.shape[-2] == 3:
        if homo:
            nvs = normalize(nvs, dim=-2)
            Ns = I - nvs @ nvs.transpose(-1, -2)
        else:
            Ns = I * (nvs.transpose(-1, -2) @ nvs) - nvs @ nvs.transpose(-1, -2)

        Ms = pts @ pts.transpose(-1, -2) @ Ns / (2 * pts.transpose(-1, -2) @ Ns @ pts)

        # TODO: Chage the parameter here. If the error gets smaller, why?
        # Mlt = w * Rs.unsqueeze(2).transpose(-1, -2) @ (Ns - 2 * Ns @ Ms) @ Rs.unsqueeze(2) # For line triangulation
        # Mlt = w * Rs.unsqueeze(2).transpose(-1, -2) @ (Ns - 2 * Ns @ Ms) @ Rs.unsqueeze(2) # For line triangulation, param changed.
        # Nlt = w * Rs.unsqueeze(2).transpose(-1, -2) @ Ns @ Rs.unsqueeze(2)
        # avg_nv = normalize(torch.sum(w * Rs.unsqueeze(2).transpose(-1, -2) @ nvs, dim=1), dim=2)

        px, dt = limb_pairs[:, 0], limb_pairs[:, 1]
        Rs = Rs.unsqueeze(2)
        D = torch.zeros((batch_size, n_views, n_joints, n_joints, 3, 3), device=device)
        w = w.view(batch_size, n_views, n_limbs, 1, 1)
        # To handle repeated proximal joints.
        px_adds = w * Rs.transpose(-1, -2) @ ((I - Ms).transpose(-1, -2) @ Ns @ (I - Ms) + Ms.transpose(-1, -2) @ Ns @ Ms) @ Rs
        for i in range(px.shape[0]):
            D[:, :, px[i], px[i]] += px_adds[:, :, i]
        D[:, :, px, dt] += -w * Rs.transpose(-1, -2) @ ((I - Ms).transpose(-1, -2) @ Ns @ Ms + Ms.transpose(-1, -2) @ Ns @ (I - Ms)) @ Rs
        D[:, :, dt, px] += -w * Rs.transpose(-1, -2) @ ((I - Ms).transpose(-1, -2) @ Ns @ Ms + Ms.transpose(-1, -2) @ Ns @ (I - Ms)) @ Rs
        D[:, :, dt, dt] += w * Rs.transpose(-1, -2) @ ((I - Ms).transpose(-1, -2) @ Ns @ (I - Ms) + Ms.transpose(-1, -2) @ Ns @ Ms) @ Rs
        D = D.transpose(3, 4).reshape(batch_size, n_views, n_joints*3, n_joints*3)

        # I = I.squeeze(2)
        # D = torch.zeros((batch_size, n_views, n_joints * 3, n_joints * 3), device=device)
        # for l in range(n_limbs):
            # px, dt = limb_pairs[l, :]
            # M = Ms[:, :, l, ...]
            # N = Ns[:, :, l, ...]
            # w_l = w[:, :, l].view(batch_size, n_views, 1, 1)
            # D[:, :, 3*px:3*px+3, 3*px:3*px+3] +=  w_l * Rs.transpose(-1, -2) @ ((I - M).transpose(-1, -2) @ N @ (I - M) + M.transpose(-1, -2) @ N @ M) @ Rs
            # D[:, :, 3*px:3*px+3, 3*dt:3*dt+3] += -w_l * Rs.transpose(-1, -2) @ ((I - M).transpose(-1, -2) @ N @ M + M.transpose(-1, -2) @ N @ (I - M)) @ Rs
            # D[:, :, 3*dt:3*dt+3, 3*px:3*px+3] += -w_l * Rs.transpose(-1, -2) @ ((I - M).transpose(-1, -2) @ N @ M + M.transpose(-1, -2) @ N @ (I - M)) @ Rs
            # D[:, :, 3*dt:3*dt+3, 3*dt:3*dt+3] +=  w_l * Rs.transpose(-1, -2) @ ((I - M).transpose(-1, -2) @ N @ (I - M) + M.transpose(-1, -2) @ N @ M) @ Rs

        return D
    else:
        # 2D limb fields
        nvs = normalize(nvs, dim=-2)
        pts = normalize(pts, dim=-2)
        nvs = torch.cat((nvs, torch.zeros(*nvs.shape[:-2], 1, 1, device=device)), dim=-2)
        ntp = nvs.transpose(-1, -2) @ pts
        npt = nvs @ pts.transpose(-1, -2)
        nnt = nvs @ nvs.transpose(-1, -2)
        ppt = pts @ pts.transpose(-1, -2)
        Mid = torch.eye(3, device=device).view(1, 1, 1, 3, 3) - (ntp * (npt + npt.transpose(-1, -2)) - nnt - ppt) / (ntp**2 - 1)

        for l in range(n_limbs):
            px, dt = limb_pairs[l, :]
            w_l = w[:, :, l].view(batch_size, n_views, 1, 1)
            D[:, :, 3*px:3*px+3, 3*px:3*px+3] +=  w_l * Rs.transpose(-1, -2) @ Mid[:, :, l] @ Rs
            D[:, :, 3*dt:3*dt+3, 3*dt:3*dt+3] +=  w_l * Rs.transpose(-1, -2) @ Mid[:, :, l] @ Rs

        return D, None, None, None
