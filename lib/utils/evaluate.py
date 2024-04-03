import torch
import numpy as np
from math import pi as PI
from lib.utils.functions import normalize


def heatmap_MSE(hm1, hm2):
    """
    calculates the difference between heatmap hm1 and hm2. Take the average as output.
    hm1, hm2: torch: batch_size x N_joint x h x w;
    """
    return 0.5 * torch.mean((hm1 - hm2)**2)


def heatmap_weighted_MSE(hm1, hm2, dtype='joint', weight=None):
    """
    calculates the weighted difference between hm1 & hm2.
    hm1, hm2: torch: batch_size x N_joint x h x w or ... x 3 x h x w;
    weight: batch_size x N_joint.
    """
    if weight is not None:
        if dtype == 'joint':
            # joint
            errs = 0.5 * torch.mean((hm1 - hm2)**2 * weight.view(*weight.shape, 1, 1), dim=(2, 3))
        elif dtype == 'limb':
            # lof
            errs = 0.5 * torch.mean((hm1 - hm2).view(*weight.shape, -1, *hm1.shape[2:])**2 * weight.view(*weight.shape, 1, 1, 1), dim=(2, 3, 4))
    else:
        errs = 0.5 * torch.mean((hm1 - hm2)**2, dim=(2, 3))
    return torch.mean(errs)


def heatmap_norm_max_dist(hm1, hm2, joint_vis=None):
    """
    The normalized distance between the maximum indices of hm1 and hm2
    hm1, hm2: torch: batch_size x N_joint x h x w;
    joint_vis: batch_size x N_joint
    """
    bs, nj, h, w, = hm1.shape
    max_idx1 = torch.argmax(hm1.view(bs, nj, -1), dim=2).float()
    x1, y1 = torch.floor(max_idx1 / w) / w, max_idx1 % w / w
    max_idx2 = torch.argmax(hm2.view(bs, nj, -1), dim=2).float()
    x2, y2 = torch.floor(max_idx2 / w) / w, max_idx2 % w / w
    if joint_vis is None:
        joint_vis = torch.ones(bs, nj, device=hm1.device)
    else:
        joint_vis = joint_vis.view(bs, nj)

    return torch.sum(torch.sqrt((x1 - x2)**2 + (y1 - y2)**2) * joint_vis) / torch.sum(joint_vis)


def vector_error(vecs1, vecs2, reduction='mean'):
    """
    vecs1, vecs2: n x dim
    """
    prods = torch.sum(vecs1 * vecs2, dim=-1) / (torch.norm(vecs1, dim=-1) * torch.norm(vecs2, dim=-1))
    # it doesn't matter it's positive or negtive
    angle_errs = torch.acos(abs(prods)) * 180 / PI
    angle_errs[torch.isnan(angle_errs)] = 0
    if reduction == 'mean':
        err = torch.mean(angle_errs)
    elif reduction == "sum":
        err = torch.sum(angle_errs)
    else:
        err = angle_errs
    return err


def line_tri_loss(ns_est, ps_est, X_gt, limb_pairs, epsilon):
    """
    ns_est: batch_size x nb x 3
    ps_est: batch_size x nb x 3
    X_gt: batch_size x nj x 3
    """
    d = 0
    for i, (px, dt) in enumerate(limb_pairs):
        d += soft_SJLD(ns_est[:, i, :], ps_est[:, i, :], X_gt[:, px, :], epsilon)
        d += soft_SJLD(ns_est[:, i, :], ps_est[:, i, :], X_gt[:, dt, :], epsilon)

    return torch.mean(d / limb_pairs.shape[0] / 2)


def soft_SJLD(ns_est, ps_est, X_gt, epsilon):
    """
    For one limb and joint. Squared Joint Limb Distance
    """
    ps_est = ps_est.unsqueeze(-1)
    ns_est = ns_est.unsqueeze(-1)
    X_gt = X_gt.unsqueeze(-1)
    N = torch.eye(3, device=ps_est.device).unsqueeze(0) - ns_est @ ns_est.transpose(-1, -2)
    d = (X_gt - ps_est).transpose(-1, -2) @ N @ (X_gt - ps_est)
    d[d > epsilon] = d[d > epsilon] ** 0.1 * epsilon ** 0.9
    return d


def dire_map_error(dm1, dm2, limb_vis=None):
    """
    dm: batch_size x nb x 3 x h x w.
    limb_vis: batch_size x nb.
    measures the direction difference between dm1 & dm2
    """
    if limb_vis is None:
        limb_vis = torch.ones(*dm1.shape[:2], 1, 1, 1, device=dm1.shape)
    else:
        limb_vis = limb_vis.view(*dm1.shape[:2], 1, 1, 1)
    sumprod = torch.sum(dm1 * dm2 * limb_vis)
    prod = torch.sum(torch.norm(dm1, dim=2) * torch.norm(dm2, dim=2) * limb_vis)

    return 1 - sumprod / prod


def dire_map_angle_err(dm1, dm2, limb_vis=None):
    """
    dm: batch_size x nb x 3 x h x w
    measures the direction difference between dm1 & dm2 in vector manner
    """
    if limb_vis is None:
        limb_vis = torch.ones(*dm1.shape[:2], device=dm1.shape)
    vec1 = normalize(torch.sum(dm1, dim=(-1, -2)), dim=2)
    vec2 = normalize(torch.sum(dm2, dim=(-1, -2)), dim=2)
    angles = torch.acos(torch.sum(vec1 * vec2, dim=2)) * limb_vis
    angles[torch.isnan(angles)] = 0

    return torch.sum(angles) / torch.sum(limb_vis) * 0.1 # in order to math the magnitude of heatmap error.


def dire_map_pixelwise_loss(gt_dire, di_map, vis):
    """
    gt_dire: batch_size x n_views x n_limbs x 3
    di_map:  batch_size x n_views x n_limbs x 3 x 64 x 64
    vis: the visulizability of limbs: batch_size x n_views x n_limbs
    """
    di_norms = torch.norm(di_map, dim=3, keepdim=True)
    gt_dire = normalize(gt_dire, dim=-1, tensor=True).view(*gt_dire.shape, 1, 1)
    loss = torch.sum((gt_dire * di_norms - di_map)**2, dim=(3, 4, 5)) / torch.sum(di_norms**2, dim=(3, 4, 5))
    # if torch.sum(vis) == 0:
    #     return 0
    # else:
        # loss = torch.sum(vis * loss) / torch.sum(vis)
    loss = torch.sum(vis * loss) / (vis.shape[0]*vis.shape[1]*vis.shape[2])
    return loss


def heatmap_SE(hm1, hm2):
    """
    Calculates the difference between heatmaps hm1 and hm2, sum the square errors.
    hm1, hm2: torch: batch_size x N_joint x h x w;
    """
    bsize = hm1.shape[0]
    return torch.sum((hm1 - hm2)**2) / bsize


def soft_SMPJPE(kps1, kps2, epsilon):
    """
    Soft Squared MPJPE by Learnable Triangulation
    kps: ... x 3
    """
    smpjpe = torch.sum((kps1 - kps2)**2, dim=-1)
    sum1 = torch.sum(smpjpe[smpjpe <= epsilon])
    sum2 = torch.sum(smpjpe[smpjpe > epsilon] ** 0.1 * epsilon ** 0.9)
    # smpjpe[smpjpe > epsilon] = smpjpe[smpjpe > epsilon] ** 0.1 * epsilon ** 0.9
    # smpjpe = torch.mean(smpjpe)
    smpjpe = (sum1 + sum2) / (smpjpe.flatten().shape[0])
    return smpjpe


def MPJPE_abs(pred_kps, gt_kps, reduction='mean'):
    """
    Calculates the ABSOLUTE MPJPE between predicted keypoints and ground truth keypoints.
    pred_kps, gt_kps: N_joints x 3
    """
    if reduction == 'mean':
        return torch.mean(torch.norm(pred_kps - gt_kps, dim=1), dim=0)
    elif reduction == 'sum':
        return torch.sum(torch.norm(pred_kps - gt_kps, dim=1), dim=0)
    else:
        return torch.norm(pred_kps - gt_kps, dim=1)


def MPJPE_rel(pred_kps, gt_kps, root_idx, reduction='mean'):
    """
    Calculates the MPJPE relative to root joint between predicted keypoints
        and ground truth keypoints.
    pred_kps, gt_kps: N_joints x 3.
    root_idx: <int> The index of the root joint.
    """
    pred_kps = pred_kps - pred_kps[root_idx:root_idx+1, :]
    gt_kps = gt_kps - gt_kps[root_idx:root_idx+1, :]
    return MPJPE_abs(pred_kps, gt_kps, reduction)


def MPLAE(pred_kps, gt_kps, limb_pairs):
    """
    pred_kps, gt_kps: batch_size x N_joints x 3
    limb_pairs: n_limbs x 2
    """
    pred_bvs = normalize(pred_kps[limb_pairs[:, 1], :] - pred_kps[limb_pairs[:, 0], :], dim=1)
    gt_bvs = normalize(gt_kps[limb_pairs[:, 1], :] - gt_kps[limb_pairs[:, 0], :], dim=1)
    angles = torch.acos(torch.sum(pred_bvs * gt_bvs, dim=1)) * 180 / PI
    angles[torch.isnan(angles)] = 0
    return torch.mean(angles)

def MPLLE(pred_kps, gt_kps, limb_pairs):
    """
    pred_kps, gt_kps: batch_size x N_joints x 3
    limb_pairs: n_limbs x 2
    """
    pred_bls = torch.norm(pred_kps[limb_pairs[:, 1], :] - pred_kps[limb_pairs[:, 0], :], dim=1)
    gt_bls = torch.norm(gt_kps[limb_pairs[:, 1], :] - gt_kps[limb_pairs[:, 0], :], dim=1)
    err_bls = torch.abs(pred_bls - gt_bls)
    return torch.mean(err_bls)

def accuracy(pred_hm, gt_kps_2d, bboxes, th):
    """
    Calculate the PCK criterion
    """
    h, w = pred_hm.shape[-2:]
    pred_hm = pred_hm.view(-1, h, w)
    gt_kps_2d = gt_kps_2d.view(-1, 2)
    pred_kps_2d, maxvals = final_preds(pred_hm, bboxes)


def max_preds(heatmaps):
    """
    heatmap: n x h x w
    """
    h, w = heatmaps.shape[1:]
    idxs = torch.argmax(heatmaps.view(-1, h*w), dim=1).unsqueeze(1)
    maxvals = torch.max(heatmaps.view(-1, h*w), dim=1).unsqueeze(1)

    preds = torch.tile(idxs, (1, 2)).float()

    # The preds are of xy order.
    preds[:, 0] = (preds[:, 0]) % w
    preds[:, 1] = torch.floor((preds[:, 1]) / w)

    pred_mask = torch.tile(torch.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.float()

    preds *= pred_mask
    return preds, maxvals


def final_preds(heatmaps, bboxes):
    coords, maxvals = max_preds(heatmaps)

    h, w = heatmaps.shape[1:]

    # post-processing
    for n in range(coords.shape[0]):
        hm = heatmaps[n]
        px = int(torch.round(coords[n][0]))
        py = int(torch.round(coords[n][1]))
        if 1 < px < w-1 and 1 < py < h-1:
            diff = np.array(
                [hm[py, px+1] - hm[py, px-1],
                 hm[py+1, px] - hm[py-1, px]]
            )
            coords[n] += np.sign(diff) * .25

    # Transform back. Bbox should be ltrb manner.
    hb, wb = bboxes[:, 3] - bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0]
    preds = bboxes[:, :2] + coords * torch.stack((wb / w, hb / h))

    return preds, maxvals