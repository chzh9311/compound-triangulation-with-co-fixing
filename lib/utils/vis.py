import numpy as np
import torch
import cv2
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
from lib.utils.functions import normalize, soft_argmax
from lib.dataset.human36m import crop_image, normalize_box

import seaborn as sns
import random


def vis_heatmap_data(image, pred_hms, gt_hms, Nj, htree, reg_method, **kwargs):
    """
    Paint keypoints and density fields on image. The overall API
    image: h x w x c
    kp_hms: N_joints x h1 x w1
    df_hms: N_bones x h2 x w2
    """
    if gt_hms is not None:
        vis_list = ["pred", "gt"]
    else:
        vis_list = ["pred"]
    cols = len(vis_list) * (2 if reg_method else 1)
    rows = Nj if not "mask" in kwargs else np.sum(kwargs["mask"])
    figsize = (rows * 5, cols * 5)
    fig = plt.figure(figsize=figsize)
    for j, name in enumerate(vis_list):
        kp_hms, df_hms = eval(name+"_hms")[:Nj, ...], eval(name+"_hms")[Nj:, ...]
        fig_pos = 0
        for i in range(Nj):
            # Joints
            if "mask" in kwargs and not kwargs["mask"][i]:
                continue
            ax = fig.add_subplot(cols, rows, (2 if reg_method else 1)*rows*j+fig_pos+1, xticks=[], yticks=[])
            fig_pos += 1
            try:
                ax.set_title(htree.node_list[i]["name"])
            except IndexError:
                ax.set_title(f"Undefined_{i}")
            draw_heatmap_on_image(ax, image, kp_hms[i])
            # Bones
            if not i == 0:
                try:
                    px, dt = htree.limb_pairs[i - 1]
                    limb_name = htree.node_list[px]["name"] + " - " + htree.node_list[dt]["name"]
                except IndexError:
                    limb_name = f"Undefined_{i}"
                if reg_method in ["heatmap2d", "vanishing_map", "openpose", "lof"]:
                    if "pred_limb_labels" in kwargs:
                        vec = kwargs[name+'_limb_labels'][i-1]
                        limb_name += " [" + ", ".join([f"{vec[i]:.2f}" for i in range(vec.shape[0])]) + "]"
                    ax = fig.add_subplot(2*len(vis_list), Nj, (2*j+1)*Nj+i+1, xticks=[], yticks=[], title=limb_name)
                    draw_heatmap_on_image(ax, image, df_hms[i-1])
                elif reg_method == "heatmap1d":
                    if name == "pred":
                        ax = fig.add_subplot(2*len(vis_list), Nj, (2*j+1)*Nj+i+1, title=limb_name)
                        gt_density = kwargs["gt_density"]
                        pred_density = kwargs["pred_density"]
                        vis_1d_densities(ax, gt_density[i-1, :], pred_density[i-1, :])
                elif reg_method == "offsetmap":
                    ax = fig.add_subplot(2*len(vis_list), Nj, (2*j+1)*Nj+i+1, xticks=[], yticks=[], title=limb_name)
                    gt_offsetmap = kwargs["gt_offsetmap"]
                    pred_offsetmap = kwargs["pred_offsetmap"]
                    draw_offset_on_image(ax, image, eval(name+"_offsetmap")[i-1, ...])
    return fig


def vis_heatmap_and_gtpts(image, pred_hms, pred_kps, gt_kps, htree, diremap=None, mask=None, info=None):
    rows = 2 + int(diremap is not None) * 4
    cols = htree.size if mask is None else (np.sum(mask) * (1 + int(diremap is not None)))
    figsize = (cols * 5, rows * 5)
    fig = plt.figure(figsize=figsize)
    fig_pos = 0
    for i in range(htree.size):
        fig_row = 0
        if mask is not None and not mask[i]:
            continue
        else:
            fig_pos += 1
        ax = fig.add_subplot(rows, cols, fig_pos, xticks=[], yticks=[])
        try:
            ax.set_title(htree.node_list[i]["name"] + info[fig_pos-1] if info is not None else "")
        except IndexError:
            ax.set_title(f"Undefined_{i}")
        draw_heatmap_on_image(ax, image, pred_hms[i])
        fig_row += 1
        ax = fig.add_subplot(rows, cols, cols*fig_row+fig_pos, xticks=[], yticks=[])
        fig_row += 1
        ax.imshow(image)
        ax.scatter(gt_kps[i, 0], gt_kps[i, 1], color='b')
        ax.scatter(pred_kps[i, 0], pred_kps[i, 1], color='r')
        if diremap is not None:
            limbs = np.where(htree.limb_pairs == i)[0]
            relatives = htree.limb_pairs[limbs, 1 - np.where(htree.limb_pairs == i)[1]]
            norm_diremap = np.linalg.norm(diremap, axis=1)
            for k, j in enumerate(limbs):
                ax = fig.add_subplot(rows, cols, cols*(fig_row+k)+fig_pos, xticks=[], yticks=[])
                draw_heatmap_on_image(ax, image, norm_diremap[j])
                ax.set_title(htree.node_list[htree.limb_pairs[j, 0]]["name"] + "-"
                             + htree.node_list[htree.limb_pairs[j, 1]]["name"])
                
                ax = fig.add_subplot(rows, cols, cols*(fig_row+k)+fig_pos+1, xticks=[], yticks=[])
                draw_heatmap_on_image(ax, image, pred_hms[relatives[k]])
                ax.set_title(htree.node_list[relatives[k]]["name"])
            fig_pos += 1
    return fig


def analyze_cofix(image, pred_hms, pred_kps, fixed_hms, fixed_kps, gt_kps, htree, pred_lfs, fixed_lfs, i, info=None):
    """co_fixing visualize

    Args:
        pred_hms (nj x h x w): predicted hms
        pred_kps (nj x 2): 
        fixed_hms (nj x h x w): _description_
        fixed_kps (nj x 2): _description_
        gt_kps (nj x 2): _description_
        pred_lfs (nb x 3 x h x w): 
        fixed_lfs (nb x 3 x h x w):
        i: jtid: the index of the drawn point.
        info (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    cols = 4
    rows = 2
    figsize = (cols * 5, rows * 5)
    fig = plt.figure(figsize=figsize)
    limbs = np.where(htree.limb_pairs == i)[0]
    for k, lfs in enumerate([pred_lfs, fixed_lfs]):
        ax = fig.add_subplot(rows, cols, k*cols + 1, xticks=[], yticks=[])
        norm_diremap = np.linalg.norm(lfs, axis=1)
        draw_multi_heatmap_on_image(ax, image, norm_diremap[limbs])
        
        # ax = fig.add_subplot(rows, cols, cols*(fig_row+k)+fig_col+1, xticks=[], yticks=[])
        # draw_heatmap_on_image(ax, image, pred_hms[relatives[k]])
        # ax.set_title(htree.node_list[relatives[k]]["name"])

    ax = fig.add_subplot(rows, cols, 2, xticks=[], yticks=[])
    try:
        ax.set_title(htree.node_list[i]["name"] + info if info is not None else "")
    except IndexError:
        ax.set_title(f"Undefined_{i}")
    draw_heatmap_on_image(ax, image, np.exp(10*pred_hms[i]))

    ax = fig.add_subplot(rows, cols, cols + 2, xticks=[], yticks=[])
    draw_heatmap_on_image(ax, image, fixed_hms[i])

    # relatives = htree.limb_pairs[limbs, 1 - np.where(htree.limb_pairs == i)[1]]
    ax = fig.add_subplot(rows, cols, 3, xticks=[], yticks=[])
    ax.imshow(image)
    htree.draw_skeleton(ax, pred_kps)
    ax.scatter(pred_kps[i, 0], pred_kps[i, 1], s=200, linewidths=3, marker='o', color=[0, 0, 0, 0], edgecolors=np.array([[1, 1, 0]]))

    ax = fig.add_subplot(rows, cols, cols + 3, xticks=[], yticks=[])
    ax.imshow(image)
    htree.draw_skeleton(ax, fixed_kps)
    ax.scatter(fixed_kps[i, 0], fixed_kps[i, 1], s=200, linewidths=3, marker='o', color=[0, 0, 0, 0], edgecolors=np.array([[1, 1, 0]]))

    ax = fig.add_subplot(rows, cols, 4, xticks=[], yticks=[])
    ax.imshow(image)
    htree.draw_skeleton(ax, gt_kps)
    plt.tight_layout()
    return fig


def analyze_compound_tri(images, pred_kps_2d, pred_di_maps, tri_kps_3d, pred_kps_3d, gt_kps_3d, htree):
    """qualitative analysis of the effect of compound triangulation.

    Args:
        images (ndarray): nv x hi x wi x 3
        pred_kps_2d (ndarray): nv x nj x 2
        pred_di_maps (ndarray): nv x nl x 3 x h x w
        tri_kps_3d (ndarray): baseline nj x 3
        pred_kps_3d (ndarray): nj x 3
        gt_kps_3d (ndarray): nj x 3
        htree: human tree
    """
    nv, nj, _ = pred_kps_2d.shape
    hi, wi = images.shape[1:3]
    cols = nv * 2 + 3
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=-np.pi/2, vmax=np.pi/2)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig = plt.figure(figsize=(cols*5, 5))
    for vi in range(nv):
        # draw skeleton
        ax = fig.add_subplot(1, cols, vi+1, xticks=[], yticks=[])
        ax.imshow(images[vi])
        htree.draw_skeleton(ax, pred_kps_2d[vi])

        # draw lof regression
        ax = fig.add_subplot(1, cols, nv + vi+1, xticks=[], yticks=[])
        ax.imshow(images[vi])
        x, y, u, v, t = get_center_vec(pred_di_maps[vi], 4, hi / pred_di_maps.shape[-1])
        ax.quiver(x, y, u, v, cmap=cmap, color=sm.to_rgba(t), angles='xy', scale=10)
    
    # cbar = plt.colorbar(sm, ax=ax, location='right', ticks=[-90, -45, 0, 45, 90])
    # cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

    for i, kps_3d in enumerate([tri_kps_3d, pred_kps_3d, gt_kps_3d]):
        ax = fig.add_subplot(1, cols, 2*nv + 1 + i, projection='3d')
        htree.draw_skeleton(ax, kps_3d)
        ax.view_init(elev=90, azim=-90)
    
    plt.tight_layout()
    
    return fig


def draw_3D_pose_with_image(ax, image, proj_mat, pose3d, htree):
    """
    pose3d is the local coordinate.
    """
    h, w = image.shape[:2]
    htree.draw_skeleton(ax, pose3d)
    yy, xx = np.meshgrid(np.arange(w), np.arange(h))


def vis_specific_vector_field(image, htree, gt_vfield, pred_vfield):
    # Randomize a vector to visualize:
    i = random.randint(0, len(htree.node_list)-2)
    px, dt = htree.limb_pairs[i]
    limb_name = htree.node_list[px]["name"] + " - " + htree.node_list[dt]["name"]
    fig = plt.figure(figsize=(80, 40))
    ax_pred = fig.add_subplot(1, 2, 1, title="predicted " + limb_name, xticks=[], yticks=[])
    draw_offset_on_image(ax_pred, image, pred_vfield[i, ...])
    ax_gt = fig.add_subplot(1, 2, 2, title="GT " + limb_name, xticks=[], yticks=[])
    draw_offset_on_image(ax_gt, image, gt_vfield[i, ...])
    return fig


def analyze_lof(image, vf, limb_ends, projection='2d'):
    """
    image: h x w x 3
    vf: 3 x h0 x w0
    limb_ends: 2 x 2
    """
    image = cv2.resize(image, (1280, 1280))
    h, w = image.shape[:2]
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=-np.pi/2, vmax=np.pi/2)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(1, 3, 1, xticks=[], yticks=[])
    # ax1.imshow(image)
    # x, y, u, v, t = get_center_vec(vf.reshape(1, *vf.shape), 20, h / vf.shape[-1])
    # ax1.quiver(x, y, u, v, cmap=cmap, color=sm.to_rgba(t), angles='xy', scale=5)
    ax2 = fig.add_subplot(1, 3, 2, xticks=[], yticks=[])
    nf = np.linalg.norm(vf, axis=0)
    draw_heatmap_on_image(ax2, image, nf)
    ax3 = fig.add_subplot(1, 3, 3, xticks=[], yticks=[], projection='3d')
    # ax3 = fig.gca(projection='3d')
    # part visualize

    # image = np.round(image / 2 + 127).astype(np.uint8)
    pad = 20
    bbox = [np.min(limb_ends[:, 0])-pad, np.min(limb_ends[:, 1])-pad, np.max(limb_ends[:, 0])+pad, np.max(limb_ends[:, 1]+pad)]
    bbox = normalize_box(bbox) * 4
    if np.max(bbox) > image.shape[0] or np.min(bbox) < 0:
        return None
    image = crop_image(image, bbox)
    lof_box = np.round(bbox/16).astype(np.uint8)
    if lof_box[3] - lof_box[1] < 25:
        stride = 1
    else:
        stride = 2
    vf = vf[:,list(range(lof_box[1], lof_box[3], stride)), :][:, :, list(range(lof_box[0], lof_box[2], stride))]
    ax1.imshow(image)

    v, u = np.meshgrid(np.arange(0, vf.shape[2]), np.arange(0, vf.shape[1]))
    u, v = u*4, v*4
    angle = np.arctan2(vf[2], np.linalg.norm(vf[:2], axis=0))
    # c = sm.to_rgba(angle)
    # ax3.quiver(u.flatten()*4*stride, v.flatten()*4*stride, vf[0], vf[1], cmap=cmap, color=sm.to_rgba(angle.flatten()), angles='xy', scale=30)
    ax3.quiver(u.flatten()*4*stride, v.flatten()*4*stride, np.zeros_like(u.flatten()),
               vf[0].flatten(), vf[1].flatten(), vf[2].flatten(), color="red", pivot='middle',
               length=5)
    extents = np.array(
        [getattr(ax3, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    ax3.auto_scale_xyz(*np.column_stack((centers - r, centers + r)))
    ax3.view_init(elev=0, azim=0)
    tmp = [getattr(ax3, 'set_{}ticks'.format(dim))([]) for dim in 'xyz']
    ax3.set_xlabel(r"$x$"); ax3.set_ylabel(r"$y$"); ax3.set_zlabel(r"$z$")
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def draw_multi_heatmap_on_image(ax, img, heatmaps):
    """
    img: h x w x 3
    heatmaps: n x h x w; non-negative
    """
    # Normalize
    img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.uint8)
    # dim the image
    # img = np.round(img / 2).astype(np.int16)

    # Conver the heatmap onto the image
    n_hm, h, w = heatmaps.shape
    heatmaps = heatmaps / (heatmaps.max(axis=(1, 2))).reshape(n_hm, 1, 1)
    # if (heatmap > 0).any() and (heatmap < 0).any():
    #     heatmap[heatmap>0] = np.round(heatmap[heatmap>0] / heatmap[heatmap>0].max() * 127)
    #     heatmap[heatmap<0] = np.round(heatmap[heatmap<0] / -heatmap[heatmap<0].min() * 127)
    # elif (heatmap > 0).any() and not (heatmap < 0).any():
    #     heatmap = np.round(heatmap / heatmap.max() * 127)
    # elif not (heatmap > 0).any() and (heatmap < 0).any():
    heatmaps = np.round(heatmaps / heatmaps.max(axis=(1, 2)).reshape(n_hm, 1, 1) * 127)
    hm_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [200, 200, 0]])

    for i in range(heatmaps.shape[0]):
        heatmap = cv2.resize(heatmaps[i].astype(np.int16), img.shape[:2]).astype(np.float32) / 255
        heatmap = heatmap.reshape((*img.shape[:2], 1))
        img = img * (1 - heatmap) + hm_colors[i].reshape(1, 1, 3) * heatmap
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)
    ax.imshow(img)


def draw_heatmap_on_image(ax, img, heatmap):
    # Normalize
    img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # Conver the heatmap onto the image
    heatmap = np.round((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
    # if (heatmap > 0).any() and (heatmap < 0).any():
    #     heatmap[heatmap>0] = np.round(heatmap[heatmap>0] / heatmap[heatmap>0].max() * 127)
    #     heatmap[heatmap<0] = np.round(heatmap[heatmap<0] / -heatmap[heatmap<0].min() * 127)
    # elif (heatmap > 0).any() and not (heatmap < 0).any():
    #     heatmap = np.round(heatmap / heatmap.max() * 127)
    # elif not (heatmap > 0).any() and (heatmap < 0).any():
    #     heatmap = np.round(heatmap / -heatmap.min() * 127)
    # cmap = mpl.cm.cool
    # min_val = heatmap.min()
    # max_val = heatmap.max()
    # hm_fig = sns.heatmap(heatmap, cmap=cmap)
    # norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val, clip=True)
    # mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # hm_fig = mapper.to_rgba(heatmap)[..., :3]
    # heatmap *= 255
    # heatmap[heatmap < 0] = 0
    # heatmap[heatmap >255] = 255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:, :, ::-1]

    heatmap = cv2.resize(heatmap, img.shape[:2])
    img = (img * 0.6 + heatmap * 0.4).astype(np.uint8)
    # img[img < 0] = 0
    img = img.astype(np.uint8)
    ax.imshow(img)


def draw_vec_field_on_image(images, di_maps):
    n_views, n_limbs, _, h, w = di_maps.shape
    di_maps = di_maps[:, :, :, list(range(0, h, 2)), :][:, :, :, :, list(range(0, w, 2))]
    fig = plt.figure(figsize=(40*n_views, 40))
    images = np.round(images / 2 + 127).astype(np.uint8)

    for i in range(n_views):
        ax = fig.add_subplot(1, n_views, i+1, xticks=[], yticks=[])
        ax.set_title(f"pred on camera {i+1}")
        ax.imshow(images[i, ...])
        # x, y, u, v, t = get_center_vec(di_maps[i], 10, h / di_maps.shape[-1], di[i, ...])
        cmap = mpl.cm.cool
        vis_limb = 0
        v, u = np.meshgrid(np.arange(0, h, 2), np.arange(0, w, 2))
        angle = np.arctan2(di_maps[i, vis_limb, 2], np.linalg.norm(di_maps[i, vis_limb, :2], axis=0))
        ax.quiver(u*4, v*4, di_maps[i, vis_limb, 0], di_maps[i, vis_limb, 1], angle, cmap=cmap, angles='xy', scale=150)
    
    return fig

def draw_di_vec_on_image(images, pred_di_maps, gt_di_maps=None, di=None):
    """
    Draw a vector derived from di_map that fits the limb prediction.
    di_maps: n_views x n_limbs x 3 x h x w
    """
    if gt_di_maps is not None:
        srclist = ["pred", "gt"]
    else:
        srclist = ["pred"]
    n_views, h, w, _ = images.shape
    fig = plt.figure(figsize=(10*n_views, 10*len(srclist)))

    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=-np.pi/2, vmax=np.pi/2)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    for s in range(len(srclist)):
        di_maps = eval(srclist[s]+"_di_maps")
        for i in range(n_views):
            ax = fig.add_subplot(len(srclist), n_views, s*n_views+i+1, xticks=[], yticks=[])
            ax.set_title(f"pred on camera {i+1}")
            ax.imshow(images[i, ...])
            # x, y, u, v, t = get_center_vec(di_maps[i], 10, h / di_maps.shape[-1], di[i, ...])
            x, y, u, v, t = get_center_vec(di_maps[i], 20, h / di_maps.shape[-1])
            ax.quiver(x, y, u, v, cmap=cmap, color=sm.to_rgba(t), angles='xy', scale=5)
        cbar = plt.colorbar(sm, ax=ax, location='right', ticks=[-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
        cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    # ax = fig.add_subplot(1, n_views+1, n_views+1, xticks=[], yticks=[])
    # plt.colorbar(sm, cax=ax)

    return fig


def get_center_vec(di_maps, l, feat_stride):
    h, w = di_maps.shape[-2:]
    n = np.linalg.norm(di_maps, axis=1, keepdims=True)
    # grids = np.stack(np.meshgrid(np.arange(w), np.arange(h), indexing='xy'), axis=0)
    ctr = soft_argmax(torch.as_tensor(n), 100).squeeze(1).numpy()[:, [1, 0]] * feat_stride
    # ctr = np.sum(grids.reshape(1, 2, h, w) * n, axis=(2, 3)) / np.sum(n, axis=(2, 3)) * feat_stride
    vec = normalize(np.sum(n * di_maps, axis=(2, 3)), dim=1, tensor=False)
    vx, vy, vz = [vec[:, i] for i in range(vec.shape[1])]
    # vx, vy, vz = [di[:, i] for i in range(3)]
    x = ctr[:, 0] - vx * l / 2
    y = ctr[:, 1] - vy * l / 2
    theta = np.arctan2(vz, np.sqrt(1-vz**2))
    return x, y, vx, vy, theta


def analyze_particular_frame(image, p2d1, p2d2, hm1, hm2, p2d_gt, lofs, indicator, htree, labels):
    fig = plt.figure(figsize=(15, 10))
    vis_joint_idx = np.where(indicator)[0][0]

    ax = fig.add_subplot(2, 3, 1, xticks=[], yticks=[])
    ax.set_title(labels[0])
    draw_heatmap_on_image(ax, image, hm1[vis_joint_idx])

    ax = fig.add_subplot(2, 3, 2, xticks=[], yticks=[])
    ax.set_title(labels[1])
    draw_heatmap_on_image(ax, image, hm2[vis_joint_idx])

    ax = fig.add_subplot(2, 3, 3, xticks=[], yticks=[])
    ax.set_title(labels[2])
    ax.imshow(image)
    htree.draw_skeleton(ax, p2d1)

    ax = fig.add_subplot(2, 3, 4, xticks=[], yticks=[])
    ax.set_title(labels[3])
    ax.imshow(image)
    htree.draw_skeleton(ax, p2d2)

    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=-np.pi/2, vmax=np.pi/2)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    limb_idctor = np.logical_or(indicator[htree.limb_pairs[:, 0]], indicator[htree.limb_pairs[:, 1]])
    lofs = lofs[limb_idctor, :]
    ax = fig.add_subplot(2, 3, 5, xticks=[], yticks=[])
    ax.set_title(labels[4])
    ax.imshow(image)
    x, y, u, v, t = get_center_vec(lofs, 20, image.shape[0] / lofs.shape[-1])
    ax.quiver(x, y, u, v, cmap=cmap, color=sm.to_rgba(t), angles='xy', scale=5)
    # cbar = plt.colorbar(sm, ax=ax, location='right', ticks=[-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
    # cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

    ax = fig.add_subplot(2, 3, 6)
    ax.set_title(labels[5])
    ax.imshow(image)
    htree.draw_skeleton(ax, p2d_gt)

    fig.subplots_adjust(wspace=0, hspace=0)

    return fig


def draw_offset_on_image(ax, image, offset):
    # Normalize
    img = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.uint8)
    # dim the image
    img = np.round(img / 2).astype(np.uint8)

    ax_img = ax.imshow(image)
    h, w, c = image.shape
    step = h / offset.shape[1]
    if offset.shape[0] == 2:
        ax.quiver(np.arange(0, w, step), np.arange(0, h, step), offset[0, :, :], offset[1, :, :], color="tab:red", angles='xy')
    else:
        # Handle with 3D vectors.
        cmap = mpl.cm.cool
        angle = np.arctan2(offset[2, ...], np.linalg.norm(offset[:2, ...], axis=0)) / np.pi * 180
        mark_norm = mpl.colors.Normalize(vmin=angle.min(), vmax=angle.max())
        print(np.sum(angle * np.linalg.norm(offset, axis=0)) / np.sum(np.linalg.norm(offset, axis=0)))
        ax.quiver(np.arange(0, w, step), np.arange(0, h, step), offset[0, :, :], offset[1, :, :], angle, units='width',
                  cmap=cmap, angles='xy', scale=10)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=mark_norm, cmap=cmap), ax=ax, location='right', shrink=0.6)
        cbar.minorticks_on()


def vis_1d_densities(ax, density1, density2):
    size = density1.shape[0]
    x = np.linspace(0, 1, size)
    ax.plot(x, density1, color="tab:blue")
    ax.plot(x, density2, color="tab:orange")
    plt.legend(["GT", "Pred"])


def vis_2d_kps(images, kps_dict, htree):
    """
    kps_dict: name -> keypoints. OrderedDict
    """
    n_views, h, w, _ = images.shape
    rows = len(kps_dict)
    fig = plt.figure(figsize=(10*n_views, 10*rows))
    for i in range(n_views):
        for j, (name, kps) in enumerate(kps_dict.items()):
            ax = fig.add_subplot(rows, n_views, j * n_views + i+1, xticks=[], yticks=[])
            if j == 0:
                ax.set_title(f"camera {i+1}")
            if i == 0:
                ax.set_ylabel(name)
            ax.imshow(images[i, ...])
            # ax.scatter(pred_kps[i, :, 0], pred_kps[i, :, 1])
            htree.draw_skeleton(ax, kps[i, ...])

    return fig


def vis_density(gt_hms, hms, gt_bvs, bvs, bvs_from_kps, gt_mus, mus):
    """
    hms: n x h x w
    bvs: n x 2 x 2
    mus: n x 1
    """
    n_vis = gt_mus.shape[0]
    figure = plt.figure(figsize=(n_vis * 4, 8))
    for i in range(n_vis):
        ax = figure.add_subplot(2, n_vis, i+1, xticks=[], yticks=[])
        ax.set_title(f"GT mu = {gt_mus[i]:.4f}")
        sns.heatmap(gt_hms[i, ...], cbar=False, xticklabels=[], yticklabels=[])
        bv = gt_bvs[i, 1, :]-gt_bvs[i, 0, :]
        l = np.linalg.norm(bv)
        ax.arrow(*gt_bvs[i, 0, :], *bv, color="b", head_width=1, head_length= 2 if l > 2.5 else 0.8 * l)
        plt.axis('equal')

        ax = figure.add_subplot(2, n_vis, n_vis + i+1, xticks=[], yticks=[])
        ax.set_title(f"mu = {mus[i]:.4f}")
        sns.heatmap(hms[i, ...], cbar=False, xticklabels=[], yticklabels=[])
        bv = bvs[i, 1, :]-bvs[i, 0, :]
        l = np.linalg.norm(bv)
        ax.arrow(*bvs[i, 0, :], *bv, color="b", head_width=1, head_length= 2 if l > 2.5 else 0.8 * l)

        bv = bvs_from_kps[i, 1, :]-bvs_from_kps[i, 0, :]
        l = np.linalg.norm(bv)
        ax.arrow(*bvs_from_kps[i, 0, :], *bv, color="g", head_width=1, head_length= 2 if l > 2.5 else 0.8 * l)
        plt.axis('equal')

    return figure


if __name__ == "__main__":
    # Test code
    # image = cv2.imread("/home/chenzhuo/Pictures/person1.jpg")
    # image = cv2.resize(image, (200, 300))
    # images = np.stack((image, image), axis=0)
    # kps = np.array([[10, 10], [40, 10], [40, 100], [60, 200]])
    # pred_kps = np.stack((kps, kps), axis=0)
    # fig = vis_2d_kps(images, pred_kps, pred_kps)
    density1 = np.abs(np.random.randn(64))
    density2 = np.abs(np.random.randn(64))
    fig = plt.figure()
    ax = fig.add_subplot()
    vis_1d_densities(ax, density1, density2)
    plt.show()