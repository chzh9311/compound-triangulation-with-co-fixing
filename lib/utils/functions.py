import os
import sys
import torch
from torch.nn import ReLU
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, floor
import seaborn as sns
import logging

def solve(A, b):
    if torch.__version__ == "1.2.0":
        return torch.solve(b, A)[0]
    else:
        return torch.linalg.solve(A, b)


def collate_pose(data):
    """
    The data output might contain array, float, int, and str.
    """
    n_types = len(data[0])
    output = [[] for k in range(n_types)]
    for d in data:
        for i in range(n_types):
            if type(d[i]) == str:
                output[i].append(d[i])
            else:
                output[i].append(torch.as_tensor(d[i]))

    output = [torch.stack(o, dim=0) if type(o[0]) != str else o for o in output]
    return output


def soft_argmax(heatmap, beta):
    """
    heatmap: ... x h x w
    equation: output = \sum_{x}\frac{exp{\beta h(x)}}{\sum_{x}exp{\beta h(x)}}x
    """

    h, w = heatmap.shape[-2:]
    heatmap = heatmap.view(*heatmap.shape[:-2], -1)
    soft_hms = F.softmax(heatmap * beta, dim=-1).view(*heatmap.shape[:-1], h, w)

    gridx, gridy = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    gridx, gridy = gridx.to(heatmap.device).float(), gridy.to(heatmap.device).float()
    norm = torch.sum(soft_hms, dim=(-1, -2))
    kp_x = torch.sum(soft_hms * gridx, dim=(-1, -2)) / norm
    kp_y = torch.sum(soft_hms * gridy, dim=(-1, -2)) / norm

    return torch.stack((kp_x, kp_y), dim=-1)


def uniform_sample(bone_density_hm, start, end, level):
    """
    bone_density_hm: bone density heatmap, batch x n_views x bone_idx x h x w
    start, end: keypoint estimations, batch x n_views x bone_idx x 2
    level: level of sample. final sample number is 2^level
    """
    # direction unit vector
    batch_size, n_views, n_bones, h, w = bone_density_hm.shape
    d = (end - start)
    l = torch.norm(d, dim=-1)
    d = d / l.unsqueeze(-1)
    device = bone_density_hm.device
    grid = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h)), dim=-1).transpose(0, 1).to(device).float()
    pos = torch.sum((grid - start.view(batch_size, n_views, n_bones, 1, 1, 2)) * d.view(batch_size, n_views, n_bones, 1, 1, 2), dim=-1)
    l = l.view(batch_size, n_views, n_bones, 1, 1)
    split_pts = [torch.zeros_like(l, device=device), l]
    sps = split_pts
    for i in range(level):
        for j in range(len(split_pts)-1):
            masked = soft_half_sign(pos - split_pts[j], 10) * soft_half_sign(split_pts[j+1] - pos, 10)
            mid = torch.sum(masked * pos, dim=(-1, -2)) / torch.sum(masked, dim=(-1, -2))
            sps.insert(2*j+1, mid.view(batch_size, n_views, n_bones, 1, 1))
        split_pts = sps.copy()
    
    split_pts = torch.stack(split_pts[1:-1], dim=-1).squeeze(-2).transpose(-1, -2)
    return start.unsqueeze(-2) + torch.matmul(split_pts, d.unsqueeze(-2))


def soft_half_sign(x, multiplier):
    """
    Based on tanh, roughly realizes:
    f(x) = 1, when x > 0; 0 when x <= 0
    """
    return F.tanh(multiplier * x)/2 + 0.5


def block_diag_batch(Ms):
    """
    do block diag calculation for the last two dims
    """
    num, h, w = Ms.shape[-3:]
    result = torch.zeros(Ms.shape[:-3] + (num*h, num*w), device=Ms.device)
    for i in range(num):
        result[..., i*h:(i+1)*h, i*w:(i+1)*w] = Ms[..., i, :, :]
    
    return result


def kronecker_prod(A, B):
    """
    do kronecker product in batches.
    Currently only dim(B) == 2 is supported.
    """
    
    result = (A.view(*A.shape, 1, 1) * B).transpose(-2, -3)
    return result.reshape(*A.shape[:-2], A.shape[-2]*B.shape[0], A.shape[-1]*B.shape[1])


def project(projection, kps_3d):
    """
    project 3D keypoints to 2D
    projection: 3 x 4;
    kps_3d: N_joints x 3
    """
    Nj = kps_3d.shape[0]
    homo_kps_2d = projection @ np.concatenate((kps_3d, np.ones((Nj, 1))), axis=1).T
    kps_2d = (homo_kps_2d[0:2, :] / homo_kps_2d[2:3, :]).T

    return kps_2d


def soft_relu(x, m):
    """
    y = ((x^2+4m)^{-1/2} + x) / 2
    """
    return (torch.sqrt((x**2 + 4 * m)) + x) / 2


def calc_vanish_from_vmap(vmap, config):
    """
    vmap: batch_size x n_bones x h x w
    config: L & FIT_METHOD
    """
    bs, nj, h, w, = vmap.shape
    max_idx = torch.argmax(vmap.view(bs, nj, -1), dim=2)
    y, x = torch.floor(max_idx / w) + 0.5, max_idx % w + 0.5
    r0 = (x - w/2) / w * 2
    theta = (y - h/2) / h * 2 * torch.pi
    rho = 2 * r0 * config.L
    w = 1 - r0 ** 2
    w[r0 < 0] *= -1
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    return torch.stack((x, y, w), dim=2)
    # pts = soft_argmax(vmap, config.SOFTMAX_BETA)[:, :, [1, 0]]


def calc_vanish(offset_map, config):
    """
    offset_map: batch_size x n_bones x 2 x h x w
    """
    bs, nb, _, h, w = offset_map.shape
    device = offset_map.device
    grids = torch.stack(torch.meshgrid([torch.arange(w), torch.arange(h)]), dim=0).transpose(1, 2).to(device).float()
    grids = grids.view(1, 1, 2, h, w)
    if config.TYPE == "lambda":
        lams = torch.norm(offset_map, dim=2, keepdim=True)
        d_infs = config.SAMPLE_LEN * (1 - lams ** 2)

        weight = 2 * lams ** 2
        offsets = torch.cat((offset_map * d_infs + grids * weight, weight), dim=2)
        pred_vpts = torch.sum(offsets, dim=(3, 4))
    else:
        mus = torch.norm(offset_map, dim=2, keepdim=True)

        weight = 1 - mus ** 2
        offsets1 = normalize(torch.cat((offset_map * 2 * config.SAMPLE_LEN + grids * weight, weight), dim=2), dim=2)
        pred_vpts1 = normalize(torch.sum(offsets1, dim=(3, 4), keepdim=True), dim=2)
        offsets2 = normalize(torch.cat((- offset_map * 2 * config.SAMPLE_LEN + grids * weight, weight), dim=2), dim=2)
        pred_vpts2 = normalize(torch.sum(offsets2, dim=(3, 4), keepdim=True), dim=2)
        indices = torch.sum(pred_vpts1 * offsets1, dim=(2, 3, 4)) > torch.sum(pred_vpts2 * offsets2, dim=(2, 3, 4))
        indices = torch.stack([indices for i in range(3)], dim=2)
        pred_vpts = pred_vpts1.squeeze() * indices + pred_vpts2.squeeze() * torch.logical_not(indices)

    return pred_vpts


def normalize(v, dim, replacenan=1, tensor=True):
    if tensor:
        n = torch.norm(v, dim=dim, keepdim=True)
        nc = n.clone()
    else:
        nc = np.linalg.norm(v, axis=dim, keepdims=True)
    nc[nc == 0] = replacenan
    return v / nc


def fit_density_simple(bdf_heatmap, bkps_estimate):
    """
    Simply use estimated keypoints to regress relative depth.
    bdf_heatmap: batch_size x n_views x n_bones x h x w.
    bkps_estimate: batch_size x n_views x n_bones x 2 x 2.
    """


def fit_density(bdf_heatmap, bvs_estimate):
    """
    bdf_heatmap: batch_size x n_views x n_bones x h x w.
    bvs_estimate: batch_size x n_views x n_bones x 2
    """
    bdf_heatmap = F.relu(bdf_heatmap)
    batch_size, n_views, n_bones, h, w = bdf_heatmap.shape

    mus = torch.zeros((batch_size, n_views, n_bones), dtype=torch.float32, device=bdf_heatmap.device)
    vis = torch.ones((batch_size, n_views, n_bones), dtype=torch.float32, device=bdf_heatmap.device)
    bvs = torch.zeros((batch_size, n_views, n_bones, 2, 2), dtype=torch.float32, device=bdf_heatmap.device)

    for batch in range(batch_size):
        for v in range(n_views):
            for b in range(n_bones):
                # 1. The middle vectors.
                heatmap = bdf_heatmap[batch, v, b, :, :]
                center, dires, vars = principal_regression(heatmap)
                n = dires[:, 0]
                var = vars[1]
                if torch.dot(n, bvs_estimate[batch, v, b, :]) < 0:
                    n = -n
                # 2. The boundaries.
                bounds, pos, dist = find_bounds(heatmap, center, n)
                # fix bounds:
                bounds[0] = -soft_relu(-(bounds[0] + torch.sqrt(var)), 0.1)
                bounds[1] = soft_relu(bounds[1] - torch.sqrt(var), 0.1)

                # 3. regresssion
                mask = torch.logical_and(pos < bounds[1], pos > bounds[0])
                mask = torch.logical_and(mask, dist**2 < 9*var)

                # tf_hm = 1 / torch.sqrt(heatmap[mask]) * torch.exp(-dist[mask] ** 2 / (4 * var))
                # tf_hm = 1 / torch.sqrt(heatmap[mask])

                if torch.sum(mask) < 16:
                    mu = 1
                    vis[batch, v, b] = 0
                else:
                    w = heatmap[mask]**2 * torch.exp(dist[mask] ** 2 / (2 * var))
                    ahat, bhat = dm_regression(pos[mask], heatmap[mask], dist[mask], var)

                    mu = ahat / (ahat + bhat * (bounds[1] - bounds[0]))

                mus[batch, v, b] = mu
                bvs[batch, v, b, 0, :] = center + n * bounds[0]
                bvs[batch, v, b, 1, :] = center + n * bounds[1]

    return mus, vis, bvs


def principal_regression_batch(heatmaps):
    """
    heatmaps: batch_size x n_views x n_joints x h x w.
    returns the most salient direction (with the largest variance).
    1. calculate the covariance of heatmap as a probability distribution;
    2. In cov, take the eigen vector of the largest eigen value.
    """
    batch_size, n_views, n_bones, h, w = heatmaps.shape
    device = heatmaps.device
    cell_grid = torch.stack(torch.meshgrid([torch.arange(w), torch.arange(h)]), dim=0).transpose(1, 2).to(device).float()
    coord_x = torch.stack([cell_grid]*batch_size * n_views).reshape(batch_size, n_views, h, w)
    coord_y = torch.stack([cell_grid]*batch_size * n_views).reshape(batch_size, n_views, h, w)

    mean_x = torch.sum(heatmaps * coord_x, dim=(3, 4)) / torch.sum(heatmaps, dim=(3, 4))
    mean_y = torch.sum(heatmaps * coord_y, dim=(3, 4)) / torch.sum(heatmaps, dim=(3, 4))

    sub_x = coord_x.view(batch_size, n_views, n_bones, h*w) - mean_x.unsqueeze(-1)
    sub_y = coord_y.view(batch_size, n_views, n_bones, h*w) - mean_y.unsqueeze(-1)

    half_cov = torch.stack(sub_x, sub_y, dim=3)
    cov = half_cov @ half_cov.transpose(-1, -2) / (h * w)

    dires = torch.zeros((batch_size, n_views, 2, 2), device=device)
    stds = torch.zeros((batch_size, n_views, 2), device=device)
    for i in range(batch_size):
        for j in range(n_views):
            for b in range(n_bones):
                ev, em = torch.eig(cov[i, j, b, ...])
                dires[i, j, b, :, :] = em.real
                stds[i, j, b, :] = ev.real
    return torch.stack((mean_x, mean_y), dim=-1), dires, stds


def principal_regression(heatmap):
    """
    heatmaps: batch_size x n_views x n_joints x h x w.
    returns the most salient direction (with the largest variance).
    """
    h, w = heatmap.shape
    device = heatmap.device
    grids = torch.stack(torch.meshgrid([torch.arange(w), torch.arange(h)]), dim=-1).to(device).float()

    mean = torch.sum(heatmap.unsqueeze(-1) * grids, dim=(0, 1)) / torch.sum(heatmap)

    half_cov = (grids.view(h*w, 2) - mean.unsqueeze(0))

    cov = (heatmap.view(h*w, 1) * half_cov).transpose(0, 1) @ half_cov / torch.sum(heatmap)

    ev, em = torch.eig(cov)
    ev, indices = torch.sort(ev.real, descending=True)
    em = em.real[:, indices]
    return mean.float(), em.float(), ev.float()


def find_bounds(heatmap, center, n):
    h, w = heatmap.shape
    sobel_filter = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                 [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype=torch.float32, device=n.device)
    grad = F.conv2d(heatmap.view(1, 1, h, w), sobel_filter.view(2, 1, 3, 3), padding=1).squeeze()
    grad = torch.sum(grad * n.view(2, 1, 1), dim=0)

    lam_up = 100
    lam_lo = -100
    # centered around the param center
    ygrid, xgrid = torch.meshgrid(torch.arange(w), torch.arange(h))
    xgrid = xgrid.to(n.device).float()
    ygrid = ygrid.to(n.device).float()
    proj = n[0] * (xgrid - center[0]) + n[1] * (ygrid - center[1])
    dist = n[1] * (xgrid - center[0]) - n[0] * (ygrid - center[1])

    # By gradient
    grad = grad.flatten()
    proj = proj.flatten()
    up_exp = F.softmax(grad * lam_up, dim=0)
    lo_exp = F.softmax(grad * lam_lo, dim=0)
    up_bd = torch.sum(up_exp * proj)
    lo_bd = torch.sum(lo_exp * proj)

    return torch.stack((lo_bd, up_bd)), proj.view(h, w), dist.view(h, w)


def dm_regression(x, hm, d, var):
    """
    x, hm: Tensor of dim n;
    d: the signed distance to the principal line.
    var: the variation.
    """
    w = hm**2 * torch.exp(d**2 / (2*var))
    wy = hm * torch.sqrt(hm) * torch.exp(d**2 / (4 * var))
    xbar = torch.sum(x * w) / torch.sum(w)
    ybar = torch.sum(wy) / torch.sum(w)
    xybar = torch.sum(x*wy) / torch.sum(w)
    x2bar = torch.sum(x**2 * w) / torch.sum(w)

    bhat = (xybar - xbar * ybar) / (x2bar - xbar**2)
    ahat = ybar - bhat * xbar

    return ahat, bhat


def fit_1d_density(density, limb_lengths):
    """
    density: batch_size x n_views x n_bones x size
    limb_lengths: batch_size x n_views x n_bones or number
    """
    bs, nv, nb, size = density.shape
    x = torch.linspace(0, 1, size, device=density.device).view(1, 1, 1, size)
    w = density ** 3
    sumw = torch.sum(w, dim=-1).unsqueeze(-1) # Normalized
    w /= sumw
    wy = density ** 2 * torch.sqrt(density) / sumw
    xbar = torch.sum(x * w, dim=-1)
    ybar = torch.sum(wy, dim=-1)
    xybar = torch.sum(x * wy, dim=-1)
    x2bar = torch.sum(w * x**2, dim=-1)

    bhat = (xybar - xbar * ybar) / (x2bar - xbar**2)
    ahat = ybar - bhat * xbar

    mu = ahat / (ahat + bhat * limb_lengths)

    return mu


def fit_mu_and_var(heatmaps, ps, ds, n_samples, reg_method="linear", output="mu"):
    """
    heatmaps: batch_size x n_views x n_bones x h x w
    ps: proximal joints of the limbs. batch_size x n_views x n_bones x 2
    ds: distal joints of the limbs. batch_size x n_views x n_bones x 2
    n_samples: how many samples inside two joints.
    """
    bs, nv, nb, h, w = heatmaps.shape
    assert n_samples >= 2
    device = heatmaps.device
    rates = torch.linspace(0, 1, n_samples, device=device).view(1, 1, 1, -1, 1)
    grids = torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device)), dim=-1).view(1, 1, 1, 1, h, w, 2).float()
    samples = (ds.unsqueeze(-2) * rates + ps.unsqueeze(-2) * (1-rates)).view(bs, nv, nb, n_samples, 1, 1, 2)
    offsets = samples - grids # bs x nv x nb x n_samples x h x w x 2
    limb_lengths = torch.norm(ds - ps, dim=-1)

    # calculate vertical variance
    bvs2d = (ds - ps) / limb_lengths.unsqueeze(-1)
    bvs2d = bvs2d[..., [1, 0]]
    bvs2d[..., 1] *= -1
    dists = torch.sum(offsets[:, :, :, 0, ...] * bvs2d.view(bs, nv, nb, 1, 1, 2), dim=-1)
    var = torch.sum(dists**2 * heatmaps, dim=(-1, -2)) / torch.sum(heatmaps, dim=(-1, -2))

    # back to calculating mu
    weights = soft_bilinear_weight(offsets)
    weights = weights[..., 0] * weights[..., 1]
    sample_values = torch.sum(heatmaps.unsqueeze(3) * weights, dim=(4, 5)) # bs x nv x nb x n_samples
    if output == "mu":
        if reg_method == "linear":
            mus = fit_1d_density(sample_values, limb_lengths)
            # assume gaussian distribution in vertical direction
        return mus, var
    else:
        return sample_values, var


def soft_bilinear_weight(x, alpha=0.001):
    """
    The extended ratio of a point to a boundary.
    """
    x0 = (1 - sqrt(1 - 4 * alpha)) / 2
    y = (1 - x**2 / (2*alpha)) * (torch.abs(x) <= x0) \
        + ((1 + x0/alpha) / 2 - x0 / alpha * torch.abs(x)) * (torch.abs(x) > x0) * (torch.abs(x) < 1 - x0) \
        + (1 - torch.abs(x))**2 / (2*alpha) * (torch.abs(x) >= 1-x0) * (torch.abs(x) < 1)
    return y


def calc_avg_direction(kps, diremaps, limb_pairs, sigma):
    """
    calculates the weighted average direction within diremaps. sigma to control the weight.
    kps: batch_size x n_views x n_joints x 2
    diremaps: batch_size x n_views x n_limbs x 3 x h x w
    limb_pairs: n_limbs x 2
    sigma: float. if < 0, no reweighting.
    """
    if sigma > 0:
        batch_size, n_views, Nb, _, h, w = diremaps.shape
        device = kps.device
        grids = torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device)), dim=-1).transpose(0, 1).view(1, 1, 1, h, w, 2).float()
        bvs = kps[:, :, limb_pairs[:, 1], :] - kps[:, :, limb_pairs[:, 0], :]
        vbvs = bvs[..., [1, 0]]
        vbvs[..., 0] *= -1
        offsets = grids - kps[:, :, limb_pairs[:, 0], :].view(batch_size, n_views, Nb, 1, 1, 2)
        dists = torch.sum(offsets * vbvs.view(batch_size, n_views, Nb, 1, 1, 2), dim=-1)
        weights = torch.exp(- dists ** 2 / (2 * sigma ** 2))
        dires = torch.sum(weights.unsqueeze(3) * diremaps, dim=(4, 5))
    else:
        norms = torch.norm(diremaps, dim=3, keepdim=True)
        dires = torch.sum(diremaps * norms, dim=(4, 5))
    return normalize(dires, dim=-1).unsqueeze(-1)


def calc_direction(proj_mats, mus=None, bones_2d=None, vanishing=None):
    """
    calculate the direction of a vector. Returns a unit vector.
    mus: batch_size x n_views x n_bones.
    bones_2d: batch_size x n_views x n_bones x 2 x 2.
    proj_mats: batch_size x n_views x 3 x 4
    reduction: how to handle with prediction results for multi-views:
        avg: take the average;
        sum: sum up.
    """
    if vanishing is None:
        bvs = bones_2d[..., 1, :] - bones_2d[..., 0, :]
        bls = torch.norm(bvs, dim=-1).unsqueeze(-1)
        n = bvs / bls

        mus = mus.unsqueeze(-1)
        # Homogeneous coordinate. batch_size x n_views x n_bones x 3
        vanish_pts = torch.cat((bones_2d[..., 0, :] * (mus - 1) + mus * bls * n, mus - 1), dim=-1)
    else:
        vanish_pts = vanishing
    KR = proj_mats[..., :3]
    d = (torch.inverse(KR).unsqueeze(2) @ vanish_pts.unsqueeze(-1))
    d = d / torch.norm(d, dim=-2).unsqueeze(-2)

    return d


def vec_selection(pts, vs, pts_new, vs_new, w, Rs, Cs, n, th1, th2):
    """
    vs / vs_new: batch_size x n_views x n_limbs x 3
    th1: minimum improve ratio;
    th2: good upper bound.
    """
    # def var(vs):
    #     avg = torch.mean(vs, dim=1, keepdim=True)
    #     return torch.mean(torch.sum((vs - avg)**2, dim=-1), dim=1)

    bins = []
    # dist_sc = []
    ang_sc = []
    for i in range(2**n):
        # binary choice vector.
        delta = torch.tensor([int((i%2**(j+1)) // 2**j) for j in range(n)], device=vs.device).view(1, n, 1, 1).float()
        bins.append(delta.squeeze())
        comb_v = delta * vs_new + (1-delta) * vs
        comb_p = delta * pts_new + (1-delta) * pts
        comb_p = torch.cat((comb_p, torch.ones(*comb_p.shape[:-1], 1, device=comb_p.device)), dim=-1)
        scs = line_set_scoring(comb_p, comb_v, Rs, Cs, w)
        # dist_sc.append(scs[0])
        ang_sc.append(scs[1])

    # dist_sc = torch.stack(dist_sc, dim=-1)
    ang_sc = torch.stack(ang_sc, dim=-1)
    # good_ = torch.logical_and(dist_sc[:, :, 0] < 10**4, ang_sc[:, :, 0] < 10**(-4))
    good_ = ang_sc[:, :, 0] < th2
    ## normalize
    # dist_sc /= dist_sc[:, :, 0:1]
    ang_sc /= ang_sc[:, :, 0:1]
    # score = (dist_sc + ang_sc) / 2
    score = ang_sc

    ## selecting
    mv, mi = torch.min(score, dim=-1)
    mi[(mv > th1) + good_] = 0
    bins = torch.stack(bins, dim=0)
    delta = bins[mi.flatten()].view(*mi.shape, n).transpose(-1, -2).unsqueeze(-1)
    return delta
    

def line_set_scoring(pts, nvs, Rs, Cs, w=None, homo=True):
    """
    Calculate the distance between limbs and estimated limb vectors.
    pts: batch_size x n_views x n_limbs x 3. The position indicator.
    nvs: batch_size x n_views x n_limbs x 3. The limb direction vector.
    Rs: batch_size x n_views x 3 x 3
    Cs: batch_size x n_views x 3 x 1. The camera centers.
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
    if homo:
        nvs = normalize(nvs, dim=-2)
        Ns = I - nvs @ nvs.transpose(-1, -2)
    else:
        Ns = I * (nvs.transpose(-1, -2) @ nvs) - nvs @ nvs.transpose(-1, -2)

    Ms = pts @ pts.transpose(-1, -2) @ Ns / (2 * pts.transpose(-1, -2) @ Ns @ pts)

    # TODO: Chage the parameter here. If the error gets smaller, why?
    # Mlt = w * Rs.unsqueeze(2).transpose(-1, -2) @ (Ns - 2 * Ns @ Ms) @ Rs.unsqueeze(2) # For line triangulation
    if w is None:
        w = torch.ones(*pts.shape[:3], 1, 1, device=device)
    else:
        w = w.view(*pts.shape[:3], 1, 1)
    Mlt = torch.sum(w * Rs.unsqueeze(2).transpose(-1, -2) @ (Ns - 2 * Ns @ Ms) @ Rs.unsqueeze(2), dim=1) # For line triangulation, param changed.
    CMlt = torch.sum(w * Rs.unsqueeze(2).transpose(-1, -2) @ (Ns - 2 * Ns @ Ms) @ (Rs @ Cs).unsqueeze(2), dim=1)
    CMC = torch.sum(w * (Rs @ Cs).unsqueeze(2).transpose(-1, -2) @ (Ns - 2 * Ns @ Ms) @ (Rs @ Cs).unsqueeze(2), dim=1)
    sc1 = (CMC - CMlt.transpose(-1, -2) @ solve(Mlt, CMlt)) / torch.sum(w, dim=1)
    sc1[sc1 <= 0] = 0.001
    Nlt = torch.sum(w * Rs.unsqueeze(2).transpose(-1, -2) @ Ns @ Rs.unsqueeze(2), dim=1)
    u, s, v = torch.svd(Nlt)
    sc2 = torch.min(s / torch.sum(w, dim=1).squeeze(-1), dim=-1)[0]

    ## Another method to measure consistency. variable.
    glimb_v = Rs.unsqueeze(2) @ nvs
    var = torch.sum(w * (glimb_v - torch.mean(glimb_v, dim=1, keepdim=True))**2, dim=(1, 3)) / torch.sum(w, dim=1).squeeze(-1)

    return sc1.squeeze(-1), var.squeeze(-1) #sc2

def sed_selection(Ps, Cs, xs, ys, n, confs0, th1, th2):
    """
    Calculate the SED of multiple view from given fundamental matrices Fs and point coordinates.
    Ps: batch_size x n_views x 3 x 4
    xs: Original points. Array of batch_size x n_views x n_joints x 2 x 1
    ys: Corrected points. Array of batch_size x n_views x n_joints x 2 x 1
    confs: the confidences of 2d joints. Array of batch_size x n_views x n_joints
    th1: minimum improve ratio;
    th2: good upper bound.
    """
    # Generate indexing
    def SED(F, x1, x2, confs):
        sum_conf = torch.sum(confs, dim=1)
        confs = confs.squeeze(-1).squeeze(-1)
        R = (x2.transpose(-1, -2) @ F @ x1).squeeze(-1).squeeze(-1)
        sed = confs * (1/torch.sum((F.transpose(-1, -2)[..., :2, :] @ x2)**2, dim=(-1, -2))
               + 1/torch.sum((F[..., :2, :] @ x1)**2, dim=(-1, -2))) * R ** 2
        return torch.sum(sed, dim=1) / sum_conf

    # Preprocess 
    KRs = Ps[..., :, :3].unsqueeze(2)
    Cs = Cs.unsqueeze(2)
    xs = torch.cat((xs, torch.ones(*xs.shape[:3], 1, 1, device=xs.device)), dim=3)
    ys = torch.cat((ys, torch.ones(*ys.shape[:3], 1, 1, device=xs.device)), dim=3)

    idxs = np.triu_indices(n, k=1)
    # torch.save(Ps @ Ps.transpose(-1, -2), os.path.join('offlines', 'pxpt.pth'))
    KR1 = KRs[:, idxs[0]]
    KR2 = KRs[:, idxs[1]]
    delta_C = Cs[:, idxs[1]] - Cs[:, idxs[0]]
    confs = confs0[:, idxs[0]] * confs0[:, idxs[1]]
    # Fs1 = cross_tensor(P2 @ C1) @ P2 @ P1.transpose(-1, -2)
    # Fs2 = P1 @ P1.transpose(-1, -2)
    Fs = cross_tensor(KR2 @ delta_C) @ KR2 @ torch.inverse(KR1)
    seds = []
    bins = []
    for i in range(2**n):
        # binary choice vector.
        delta = torch.tensor([int((i%2**(j+1)) // 2**j) for j in range(n)], device=xs.device).view(1, n, 1, 1, 1).float()
        bins.append(delta.squeeze())
        comb_x = delta * ys + (1-delta) * xs

        x1 = comb_x[:, idxs[0]]
        x2 = comb_x[:, idxs[1]]
        seds.append(SED(Fs, x1, x2, confs))
    
    seds = torch.stack(seds, dim=-1)
    mv, mi = torch.min(seds, dim=-1)
    mi[(mv > th1 * seds[:, :, 0]) + (seds[:, :, 0] < th2)] = 0
    bins = torch.stack(bins, dim=0)
    delta = bins[mi.flatten()].view(*mi.shape, n).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
    comb_x = delta * ys + (1-delta) * xs
    return comb_x[..., :2, :]


def heatmap_std(hm, di=False):
    """
    hm: ... x h x w.
    di=True: ... x c x h x w. Keeps direction
    """
    if di:
        hm = torch.norm(hm, dim=-3)
    std = torch.sqrt(torch.mean((hm - torch.mean(hm, dim=(-1, -2), keepdim=True))**2, dim=(-1, -2)))
    return std


def heatmap_normalize(hm, std, di=False):
    """
    hm: ... x h x w
    std: ...
    mean is not needed.
    """
    prev_std = heatmap_std(hm, di).unsqueeze(-1).unsqueeze(-1)
    std = std.unsqueeze(-1).unsqueeze(-1)
    if di:
        std = std.unsqueeze(-1)
        prev_std = prev_std.unsqueeze(-1)
    return hm * std / prev_std


def cross_tensor(vec):
    """
    vec: ... x 3 x 1.
    return [vec]x
    """
    vec = vec.squeeze(-1)
    mat = torch.zeros(*vec.shape, 3, device=vec.device)
    mat[..., [2, 0, 1], [1, 2, 0]] = vec
    mat[..., [1, 2, 0], [2, 0, 1]] = -vec
    return mat


def bilinear_line_integral_offline(hm_shape):
    """
    calculate the constant vector params for line integral for heatmap of shape hm_shape
    """
    h, w = hm_shape
    # Calculate the vector from the upper-left to all points.
    def intersections(grid_tl, end):
        x, y = end
        # ax + by + c = 0; a=y, b=-x, c=0
        isec = []
        grids = [grid_tl, (grid_tl[0]+1, grid_tl[1]),
                 (grid_tl[0]+1, grid_tl[1]+1), (grid_tl[0], grid_tl[1]+1)]
        indicators = [grids[i][0] * y - grids[i][1] * x for i in range(4)]
        if indicators[0] * indicators[1] < 0:
            isec.append((grid_tl[1] * x / y, grid_tl[1]))
        if indicators[1] * indicators[2] < 0:
            isec.append((grid_tl[0] + 1, (grid_tl[0]+1) * y / x))
        if indicators[2] * indicators[3] < 0:
            isec.append(((grid_tl[1]+1) * x / y, grid_tl[1]+1))
        if indicators[3] * indicators[0] < 0:
            isec.append((grid_tl[0], grid_tl[0] * y / x))
        for i in range(4):
            if indicators[i] == 0:
                isec.append(grids[i])
        return isec

    def line_int_values(pt1, pt2, u, v):
        x1, y1 = pt1
        x2, y2 = pt2
        x1 -= u
        x2 -= u
        y1 -= v
        y2 -= v
        s = sqrt((x2-x1)**2 + (y2-y1)**2)
        p11 = (x2 - x1) * (y2 - y1) / 3 + (x1 * (y2 - y1) + y1 * (x2 - x1)) / 2 + x1 * y1
        p10 = -((x2 - x1) * (y2 - y1) / 3 + ((x1-1) * (y2 - y1) + y1 * (x2 - x1)) / 2 + (x1-1) * y1)
        p01 = -((x2 - x1) * (y2 - y1) / 3 + (x1 * (y2 - y1) + (y1-1) * (x2 - x1)) / 2 + x1 * (y1-1))
        p00 = (x2 - x1) * (y2 - y1) / 3 + ((x1-1) * (y2 - y1) + (y1-1) * (x2 - x1)) / 2 + (x1-1) * (y1-1)
        return np.array([[p00, p10], [p01, p11]]) * s
        
    element0 = np.zeros((h, w, 2, h, w))
    for y in tqdm(range(h)):
        for x in range(w):
            if x > 0 or y > 0:
                vec = np.array([x, y])
                for v in range(y):
                    for u in range(x):
                        isecs = intersections((u, v), (x, y))
                        assert len(isecs) <= 2
                        if len(isecs) == 2:
                            element0[y, x, :, v:v+2, u:u+2] += vec.reshape(2, 1, 1) \
                                * line_int_values(isecs[0], isecs[1], u, v).reshape(1, 2, 2) / sqrt(x**2 + y**2)
    # element1 = element0[:, :, :, :, ::-1]

    # return sparse.csr_matrix(element0.reshape(h*w, 2*h*w)), sparse.csr_matrix(element1.reshape(h*w, 2*h*w))
    return element0


def euler2rot(a, b, c, order='xyz'):
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    if order == "xyz":
        return rx @ ry @ rz
    elif order == 'rpy':
        return rz @ ry @ rx


def linear_eigen_method_pose(n_cams, Xs, Ps, confidences=None):
    """
    linear eigen triangulation method for the whole human pose.
    :n_cams:      <int> the number of cameras
    :Xs:          <numpy.ndarray> of n_camera x n_joint x 2. The 2D pose estimations.
    :Ps:          <numpy.ndarray> of n_camera x 3 x 4. The camera reprojection matrices.
    :confidences: <numpy.ndarray> of n_camera x n_joint. The confidences for each
        joint on each view.
    return:       <numpy.ndarray> of n_joint x 3. The 3D joint position estimations.
    """
    n_joints = Xs.shape[1]
    linear_X = []
    if confidences is None:
        confidences = np.ones((n_cams, n_joints)) / n_cams
    for i in range(n_joints):
        linear_X.append(linear_eigen_method(n_cams, Xs[:, i, :],
            Ps, confidences[:, i]).reshape(3,))
    return np.stack(linear_X, axis=0)


def linear_eigen_method(n_cams, Xs, Ps, confidences=None):
    """
    linear eigen triangulation method for the whole human pose.
    :n_cams:      <int> the number of cameras
    :Xs:          <numpy.ndarray> of n_camera x 2. The 2D pose estimations.
    :Ps:          <numpy.ndarray> of n_camera x 3 x 4. The camera reprojection matrices.
    :confidences: <numpy.ndarray> of n_camera. The confidences for each view.
    return:       <numpy.ndarray> of 3. The 3D joint position estimations.
    """
    A_rows = []
    if confidences is None:
        confidences = np.ones(n_cams)
    for i in range(n_cams):
        conf = confidences[i]
        A_rows += [(Xs[i, 0] * Ps[i, 2:3, :] - Ps[i, 0:1, :])*conf, (Xs[i, 1] * Ps[i, 2:3, :] - Ps[i, 1:2, :])*conf]

    A = np.concatenate(tuple(A_rows), 0)
    ev, em = np.linalg.eig(A.T @ A)
    sln = em[:, np.argmin(ev):np.argmin(ev)+1]
    sln = sln[0:3, :] / sln[3]
    return sln


if __name__ == "__main__":
    a = torch.randn(3, 5, 3)
    b = torch.eye(3)
    c = kronecker_prod(a, b)
    sns.heatmap(c[0, ...])
    plt.show()
