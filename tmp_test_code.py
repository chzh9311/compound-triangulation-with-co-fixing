import pickle
import os
import sys
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms

from tqdm import tqdm
from collections import OrderedDict

from easydict import EasyDict as edict

from lib.utils.functions import *
from lib.utils.evaluate import vector_error
from lib.utils.DictTree import create_human_tree
from lib.utils.vis import vis_density, vis_specific_vector_field, draw_di_vec_on_image, vis_heatmap_data, draw_heatmap_on_image, vis_2d_kps, analyze_particular_frame, analyze_lof
from lib.dataset.human36m import Human36MMonocularFeatureMapDataset, generate_vanishing_map, Human36MMultiViewDataset
from lib.dataset.totalcapture import TotalCaptureMonocularFeatureMapDataset, TotalCaptureMultiViewDataset
from lib.models.MV_field_pose_net import optimize_wrt_params, MultiViewFPNet
from lib.dataset.mhad import MHADHeatmapDataset
from lib.dataset.joint import build_2D_dataset, JointDataset
from lib.models.layers import rdsvd
from config import get_config

import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
 

import logging

logger = logging.Logger("Hello")
logger.setLevel(logging.INFO)


def test():
    with open(os.path.join("tmp_npy_files", "debug_data1115.pkl"), "rb") as f:
        data = edict(pickle.load(f))

    idx = 4
    gt_bvs = data.gt_bvs[:, 1, :] - data.gt_bvs[:, 0, :]
    format = logging.Formatter("%(asctime)s - %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(format)
    logger.addHandler(sh)
    logger.info("data loaded")
    # th = 0.01
    # data.dm[data.dm<th] = 0
    # data.dm[idx, ...] = F.softmax(data.dm[idx, ...].flatten() * 40).view(64, 64)
    # fit_density(F.relu(data.dm[idx, ...].reshape(1, 1, 1, 64, 64).float()), gt_bvs[idx, ...].reshape(1, 1, 1, 2).float())
    # print(dire_var(F.relu(data.dm[idx, ...].reshape(64, 64).float()).detach().cpu(), torch.tensor([0.1257, -0.9921])))
    # data.dm = F.softmax(data.dm.flatten()).reshape(5, 64, 64)
    mus, bone_vis, bvs = fit_density(F.relu(data.gt_dm.reshape(1, 1, 5, 64, 64).float()), gt_bvs.reshape(1, 1, 5, 2).float())
    # fig = plt.figure(figsize=(20, 8))
    mus = mus.squeeze().detach().cpu().numpy()
    bvs = bvs.squeeze().detach().cpu().numpy()
    for key in data.keys():
        data[key] = data[key].detach().cpu().numpy()
    fig = vis_density(data.gt_dm, data.gt_dm, data.gt_bvs, bvs, data.bvs_from_kps, data.gt_mus, mus)
    plt.show()
    # for i in range(5):
    #     ax1 = fig.add_subplot(2, 5, i+1)
    #     sns.heatmap(data.gt_dm[i, ...].detach().cpu().numpy(), xticklabels=[], yticklabels=[])
    #
    #     ax2 = fig.add_subplot(2, 5, 5+i+1)
    #     sns.heatmap(data.dm[i, ...].detach().cpu().numpy(), xticklabels=[], yticklabels=[])
    #
    # plt.show()


def dire_var(heatmap, di):
    h, w = heatmap.shape
    device = heatmap.device
    di = di / torch.norm(di)
    grids = torch.stack(torch.meshgrid([torch.arange(w), torch.arange(h)], indexing="xy"), dim=-1).to(device)

    mean = torch.sum(heatmap.unsqueeze(-1) * grids, dim=(0, 1)) / torch.sum(heatmap)
    error = torch.sum((grids.view(h*w, 2) - mean.view(1, 2)) * di.view(1, 2), dim=1)
    var = torch.sum(heatmap.flatten() * error ** 2) / torch.sum(heatmap)

    return var


def file_watch():
    cfg = get_config("hourglass/joint-hourglass-backbone")
    fp = cfg.DATASET.H36M_LABELS
    labels = np.load(fp, allow_pickle=True).item()
    print(labels["cameras"][0, 0]["K"])


def fun_test():
    htree = create_human_tree()
    # np.random.seed(0)
    # forward
    # cfg = edict({"SAMPLE_LEN": 64, "TYPE": "mu"})
    hm_size = 64
    cfg = edict({"L": 256, "FIT_METHOD": "rt"})
    np.random.seed(0)
    mus = np.random.rand(16) * 0.1 + 0.95
    print(mus)
    joints = np.random.rand(17, 2) * hm_size / 2
    bones = joints[htree.limb_pairs, :]
    bvs = bones[:, 1, :] - bones[:, 0, :]
    vanish_map, vns = generate_vanishing_map((hm_size, hm_size), mus, bones, cfg.L, "rt", 2)
    # offset_map = torch.as_tensor(offset_map)
    pred_vns = calc_vanish_from_vmap(torch.as_tensor(vanish_map).unsqueeze(0), cfg).squeeze()
    pred_vns /= torch.norm(pred_vns, dim=1, keepdim=True)
    print(vector_error(pred_vns, torch.as_tensor(vns)))
        # sns.heatmap(vanish_map[i, ...])
        # plt.show()
    # lam = torch.norm(offset_map, dim=1)
    # pred_vanish = calc_vanish(offset_map.view(1, *offset_map.shape), cfg).squeeze()
    # print(torch.sum(vns * pred_vanish, dim=1) / (torch.norm(vns, dim=1) * torch.norm(pred_vanish, dim=1)))
    #
    # for i in range(16)
    #     plt.quiver(np.arange(0, 64), np.arange(0, 64), offset_map[i, 0, :, :], offset_map[i, 1, :, :], color="tab:red", angles='xy')
    #     plt.arrow(bones[i, 0, 0], bones[i, 0, 1], bvs[i, 0], bvs[i, 1], color="tab:blue")
    #     plt.scatter(vns[i, 0]/vns[i, 2], vns[i, 1]/vns[i, 2])
    #     plt.scatter(pred_vanish[i, 0]/pred_vanish[i, 2], pred_vanish[i, 1]/pred_vanish[i, 2])
    #     plt.show()


def case_study():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = get_config("ResNet152/totalcapture-ResNet152-320x320")
    # cfg = get_config("ResNet152/totalcapture-ResNet152-320x320-singlebranch")
    # cfg = get_config("experiments/ResNet152/human3.6m-ResNet152-384x384.yaml")
    cfg.DATASET.WITH_DAMAGED_ACTIONS = False
    # cfg = get_config("ResNet152/human3.6m-ResNet152-singlebranch")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if cfg.DATASET.NAME == 'human3.6m':
        test_set = Human36MMultiViewDataset(
            root_dir=cfg.DATASET.H36M_ROOT,
            label_dir=cfg.DATASET.H36M_LABELS,
            image_shape=tuple(cfg.MODEL.IMAGE_SIZE),
            is_train=False,
            transform=transform,
            crop=True,
            with_damaged_actions=cfg.DATASET.WITH_DAMAGED_ACTIONS,
            output_type=cfg.MODEL.REQUIRED_DATA,
            sigma=cfg.MODEL.EXTRA.SIGMA
        )
    else:
        test_set = TotalCaptureMultiViewDataset(
            root_dir=cfg.DATASET.TC_ROOT,
            label_dir=cfg.DATASET.TC_LABELS,
            image_shape=tuple(cfg.MODEL.IMAGE_SIZE),
            is_train=False,
            transform=transform,
            crop=True,
            output_type=cfg.MODEL.REQUIRED_DATA,
            use_cameras=cfg.TEST.USE_CAMERAS,
            frame_sample_rate=cfg.TEST.FRAME_SAMPLE_RATE,
            sigma=cfg.MODEL.EXTRA.SIGMA
        )
    # test_set = TotalCaptureMonocularFeatureMapDataset(
    #     root_dir=cfg.DATASET.TC_ROOT,
    #     label_dir=cfg.DATASET.TC_ROOT,
    #     sigma=cfg.MODEL.EXTRA.SIGMA,
    #     image_shape=tuple(cfg.MODEL.IMAGE_SIZE),
    #     heatmap_shape=tuple(cfg.MODEL.EXTRA.HEATMAP_SIZE),
    #     output_type=cfg.MODEL.REQUIRED_DATA,
    #     use_cameras=[1, 3, 5, 7],
    #     is_train=False,
    #     transform=transform,
    #     crop=True
    # )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS
    )
    for batch_i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        required_data = edict()
        for d_i, out in enumerate(config.MODEL.REQUIRED_DATA):
            if out in ["identity", "index", "subject"]:
                # data_id
                required_data[out] = data[d_i]
            else:
                required_data[out] = data[d_i].to(device).float()

    human_tree = create_human_tree(cfg.DATASET.NAME)
    mpl.use("TkAgg")
    Nj, Nb = cfg.MODEL.NUM_JOINTS, cfg.MODEL.NUM_BONES
    # vis_idx = 12
    od = np.arange(2000)
    # np.random.shuffle(od)
    # for vis_idx in range(, 2000):
    # vis_idx = 0
    vis_idx = 64
    data = test_set[vis_idx]

    required_data = edict()
    model_out = edict()
    for i, out in enumerate(cfg.MODEL.REQUIRED_DATA):
        if out == "identity":
            required_data[out] = data[i]
        else:
            required_data[out] = torch.as_tensor(data[i]).to(device).float().unsqueeze(0)

    ### study camera poses.
    # Rs = required_data.rotation.squeeze().detach().cpu().numpy()
    # nv = Rs.shape[0]
    # Cs = required_data.cam_ctr.squeeze().detach().cpu().numpy()
    # gt_p3d = required_data.keypoints3d.squeeze().detach().cpu().numpy()
    # baselines = np.zeros((nv, nv))
    # angles = np.zeros((nv, nv))
    # for i in range(nv):
    #     vec1 = Rs[i, 2, :]
    #     for j in range(i+1, nv):
    #         baselines[i, j] = np.linalg.norm(Cs[i] - Cs[j])
    #         vec2 = Rs[j, 2, :]
    #         angles[i, j] = np.arccos(vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))) * 180 / np.pi
    # np.savetxt(os.path.join("offlines", f"{cfg.DATASET.NAME}_baselines.csv"), baselines, delimiter=',')
    # np.savetxt(os.path.join("offlines", f"{cfg.DATASET.NAME}_angles.csv"), angles, fmt='%.2f', delimiter=',')
    # fig = vis_cam_and_human_pose(human_tree, gt_p3d, Rs, Cs)
    # plt.show()
    # return
    
    ### study case
    model = MultiViewFPNet(cfg, is_train=False).to(device)
    model.backbone.load_backbone_params(cfg.MODEL.BACKBONE_WEIGHTS, load_confidences=cfg.MODEL.USE_CONFIDENCE)
    model.eval()

    if 'cam_ctr' in required_data:
        out_values = model(required_data.images, required_data.projections, human_tree, required_data.intrinsics, rotation=required_data.rotation, cam_ctr=required_data.cam_ctr)
    else:
        out_values = model(required_data.images, required_data.projections, human_tree, required_data.intrinsics, rotation=required_data.rotation)

    for i, k in enumerate(cfg.MODEL.MODEL_OUTPUT):
        model_out[k] = out_values[i]

    n_limbs, n_joints, n_views = cfg.MODEL.NUM_BONES, cfg.MODEL.NUM_JOINTS, 4
    images = required_data.images[0, ...].detach().cpu().numpy()
    images = np.stack([images[:, 2-i, ...] for i in range(images.shape[1])], axis=3)
    images = np.round((images - images.min()) / (images.max() - images.min()) * 255).astype(np.uint8)

    gt_kps_2d = np.zeros((4, cfg.MODEL.NUM_JOINTS, 2))
    gt_p3d = required_data.keypoints3d.squeeze().detach().cpu().numpy()
    projection = required_data.projections.squeeze().detach().cpu().numpy() # 4 x 3 x 4
    for i in range(gt_kps_2d.shape[0]):
        P = projection[i, ...]
        homo_kp_3d = np.concatenate((gt_p3d.T, np.ones((1, gt_p3d.shape[0]))), axis=0)
        homo_kp_2d = P @ homo_kp_3d
        gt_kps_2d[i, ...] = (homo_kp_2d[:2, :] / homo_kp_2d[2:3, :]).T
    pred_kps_2d = model_out.keypoints2d.squeeze().detach().cpu().numpy()
    hms = model_out.heatmap.squeeze(0).detach().cpu().numpy()
    dms = model_out.lof.squeeze(0).detach().cpu().numpy()
    # pred_limb_labels = normalize(np.sum(dms, axis=(2, 3)), dim=1, tensor=False)

    # vis_idx
    vis_idx = 0
    images = required_data.images[vis_idx, ...].detach().cpu().numpy()
    images = np.stack([images[:, 2-i, ...] for i in range(3)], axis=3)
    images = np.round((images - images.min()) / (images.max() - images.min()) * 255).astype(np.uint8)
    pred_kps_2d = model_out.keypoints2d[vis_idx, ...].detach().cpu().numpy() # 4 x 17 x 2
    gt_kps_3d = required_data.keypoints3d[vis_idx, ...].detach().cpu().numpy() # 17 x 3
    projection = required_data.projections[vis_idx, ...].detach().cpu().numpy() # 4 x 3 x 4

    # Fixed
    # fixed_kps_2d = model_out.keypoints2d_fixed.squeeze(-1)[vis_idx, ...].detach().cpu().numpy()
    # combined_kps_2d = model_out.keypoints2d_combined.squeeze(-1)[vis_idx, ...].detach().cpu().numpy()
    # pred_p3d = model_out.keypoints3d_combined.squeeze().detach().cpu().numpy()
    # print(np.mean(np.linalg.norm(gt_p3d - pred_p3d, axis=-1)))
    # fixed_lb_dm = model_out.lof_combined[vis_idx, :, :cfg.MODEL.NUM_BONES, ...].detach().cpu().numpy()

    gt_kps_2d = np.zeros((pred_kps_2d.shape[0], cfg.MODEL.NUM_JOINTS, 2))
    for i in range(gt_kps_2d.shape[0]):
        P = projection[i, ...]
        homo_kp_3d = np.concatenate((gt_kps_3d.T, np.ones((1, gt_kps_3d.shape[0]))), axis=0)
        homo_kp_2d = P @ homo_kp_3d
        gt_kps_2d[i, ...] = (homo_kp_2d[:2, :] / homo_kp_2d[2:3, :]).T

    # pred_lb_dm = model_out.lof[vis_idx, :, :cfg.MODEL.NUM_BONES, ...].detach().cpu().numpy()
    # for cam_idx in range(4):
    #     pred_lb_dm_cam = np.linalg.norm(pred_lb_dm[cam_idx, ...], axis=1)
    #     pred_hm = model_out.heatmap #.view(required_data.images.shape[:2] + model_out.heatmap.shape[1:])
    #     pred_hm = pred_hm[vis_idx, cam_idx, :cfg.MODEL.NUM_JOINTS].detach().cpu().numpy()

    #     pred_hm = np.concatenate((pred_hm, pred_lb_dm_cam), axis=0)
        # fig = vis_heatmap_data(images[cam_idx], pred_hm, None, cfg.MODEL.NUM_JOINTS, human_tree, "heatmap2d"),

    # fig = vis_2d_kps(images, OrderedDict({"pred": pred_kps_2d, "fixed": fixed_kps_2d,
    #                                     "combined": combined_kps_2d, "gt": gt_kps_2d}), human_tree),
    # plt.show()

    # Case study: frame 864, view 3
    vis_idx = 2
    confs = model_out.confidences[0, vis_idx].detach().cpu().numpy()
    ## for double branches
    hms = hms[vis_idx] * confs[Nb:].reshape(Nj, 1, 1)
    ## for single branch
    # hms = hms[vis_idx] * confs.reshape(Nj, 1, 1)

    image = images[vis_idx]
    LFoot = hms[11]
    LKnee = hms[12]

    dms = dms[vis_idx] * confs[:Nb].reshape(Nb, 1, 1, 1)
    # LFoot = np.exp(20*LFoot)
    # LKnee = np.exp(20*LKnee)
    LFoot -= np.mean(LFoot)
    LKnee -= np.mean(LKnee)
    LK2LF = dms[10, :2]
    h, w = LFoot.shape
    coords = np.stack(np.meshgrid(np.arange(2*h-1), np.arange(2*w-1), indexing='xy'), axis=0)
    centre = np.array([h, w]).reshape(2, 1, 1)
    ker = centre - coords
    dist = np.linalg.norm(ker, axis=0)
    dist[h, w] = 1
    ker1 = ker / dist ** 2
    # ker1 = ker1 / (h*w-1)
    ker2 = ker1
    # ker2 = ker / dist **1.5
    LFoot[LFoot < 0] = 0
    LKnee[LKnee < 0] = 0
    Hs = LFoot - LKnee
    # Hs = np.zeros((h, w))
    # Hs[20, 20] = 100
    # Hs[60, 60] = -100
    convs = np.zeros((2, h, w))
    cords = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='xy'), axis=0)
    for i in range(h):
        for j in range(w):
            # convolution with reversed kernel
            convs[:, i, j] = np.sum(Hs[::-1, ::-1].reshape(1, h, w) * ker1[:, i:i+h, j:j+w], axis=(1, 2))
    
    out_dir = os.path.join('result_analysis', 'co_fix_analysis', 'sample')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    def save_drawing_fig(name, out_dir=out_dir):
        plt.savefig(os.path.join(out_dir, name), bbox_inches='tight', pad_inches=0)

    sp_ker = ker1[:, ::2, ::2]
    fig = plt.figure(figsize=(5, 5))

    # 3D field plot
    plt.imsave(os.path.join(out_dir, 'image.png'), image)
    LK2LF3d = dms[10, :, 16:36, 26:46]
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], zticks=[], projection='3d')
    tmp_coords = coords[:, 16:36, 26:46]
    ax.quiver(tmp_coords[0, ::2, ::2].flatten(), tmp_coords[1, ::2, ::2].flatten(), 0, LK2LF3d[0, ::2, ::2].flatten(),
              LK2LF3d[1, ::2, ::2].flatten(), LK2LF3d[2, ::2, ::2].flatten(), length=600, normalize=False)
    ax.set_zlim([-15, 5])
    plt.show()

    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax.quiver(cords[0].flatten(), cords[1].flatten(), sp_ker[0].flatten(), sp_ker[1].flatten(), scale=5, width=0.005)
    ax.axis("equal")
    ax.axis('off')
    save_drawing_fig('kernel.png')

    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax.quiver(cords[0, ::2, ::2].flatten(), cords[1, ::2, ::2].flatten(),
              convs[0, ::2, ::2].flatten(), convs[1, ::2, ::2].flatten(), scale=10)
    ax.axis("equal")
    ax.axis('off')
    save_drawing_fig('convolved.png')

    dot_prod = np.sum(convs * LK2LF, axis=0)
    dot_prod[dot_prod < 0] = 0
    LK2LF_new = LK2LF * dot_prod.reshape(1, h, w)
    # LK2LF_new = LK2LF + convs

    lof2jt = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            lof2jt[h-i-1, w-j-1] = np.sum(LK2LF_new * ker2[:, i:i+h, j:j+w])
        
    dist_jh = lof2jt.copy()
    prox_jh = -lof2jt.copy()
    prox_jh[lof2jt > 0] = 0
    dist_jh[lof2jt < 0] = 0
    LKnee_new = LKnee * prox_jh
    LFoot_new = LFoot * dist_jh

    cbar_kws = {'location': "top", 'format': '%.0e'}
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    sns.heatmap(LFoot, xticklabels=False, yticklabels=False, cbar=True, cmap=mpl.cm.coolwarm, cbar_kws=cbar_kws)
    ax.axis('off')
    ax.axis('equal')
    save_drawing_fig('LFoot_hm.png')

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    sns.heatmap(LKnee, xticklabels=False, yticklabels=False, cbar=True, cmap=mpl.cm.coolwarm, cbar_kws=cbar_kws)
    ax.axis('equal')
    ax.axis('off')
    save_drawing_fig('LKnee_hm.png')

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    sns.heatmap(Hs, xticklabels=False, yticklabels=False, cbar=True, cmap=mpl.cm.coolwarm, cbar_kws=cbar_kws)
    ax.axis('equal')
    plt.axis('off')
    save_drawing_fig('Subtract_hm.png')

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    sns.heatmap(lof2jt, xticklabels=False, yticklabels=False, cbar=True, cmap=mpl.cm.coolwarm, cbar_kws=cbar_kws)
    ax.axis('equal')
    plt.axis('off')
    save_drawing_fig('lof2jt_norm.png')
    # for i in range(4):
    #     eval(f"ax{i}").axis('equal')

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    draw_heatmap_on_image(ax, image, LFoot)
    ax.axis('equal')
    plt.axis('off')
    save_drawing_fig('LFoot_before.png')

    plt.cla()
    draw_heatmap_on_image(ax, image, np.linalg.norm(LK2LF, axis=0))
    plt.axis('off')
    save_drawing_fig('LK2LF_before.png')

    plt.cla()
    draw_heatmap_on_image(ax, image, LKnee)
    plt.axis('off')
    save_drawing_fig('LKnee_before.png')

    plt.cla()
    draw_heatmap_on_image(ax, image, LFoot_new)
    plt.axis('off')
    save_drawing_fig('LFoot_new.png')
    plt.cla()
    draw_heatmap_on_image(ax, image, np.linalg.norm(LK2LF_new, axis=0))
    plt.axis('off')
    save_drawing_fig('LK2LF_new.png')
    plt.cla()
    LK2LF_new *= 10e4
    ax.quiver(cords[0, ::2], cords[1, ::2], LK2LF_new[0, ::2].flatten(), LK2LF_new[1, ::2].flatten(), scale=30)
    ax.axis("equal")
    ax.axis('off')
    save_drawing_fig('convolved_new.png')
    plt.cla()
    draw_heatmap_on_image(ax, image, LKnee_new)
    plt.axis('off')
    save_drawing_fig('LKnee_new.png')
    # plt.show()


def regularize_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('equal')
    return ax


def generate_interpolate_map(start, end, h, w):
    vec = end - start
    ang = np.atan2(vec[1], vec[0])
    vvec = np.array([[0, 1], [-1, 0]]) @ vec
    coords = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='xy'), axis=2)
    offsets = coords - start.reshape(1, 1, 2)
    dist = vvec * offsets


def find_max_pts(heatmap, k=15, th=0.1):
    """
    heatmap: h x w
    return : the region(s) with the valid joint response.
    """
    # normalize
    h, w = heatmap.shape
    result = []
    heatmap = np.exp(heatmap * 40)
    heatmap /= np.sum(heatmap, axis=(-1, -2), keepdims=True)
    while True:
        indicator = conv2d(heatmap, np.ones((k, k)))
        max_xy = np.argmax(indicator)
        y, x = int(max_xy / indicator.shape[1]), int(max_xy % indicator.shape[1])
        max_value = indicator[y, x]
        if max_value > th:
            result.append((x, y))
            # heatmap[max(y-int(k/2), 0):min(y+int(k/2), h), max(x-int(k/2), 0):min(x+int(k/2), w)] = 0
            heatmap[y:y+k, x:x+k] = 0
            # th = max(max_value * 0.5, 0.01)
        else:
            break
    
    return result
    
    # indicator = indicator > 0.3
    # coords = np.stack(np.where(indicator), axis=0)
    # counted_point_set = []
    # for i in range(coords.shape[0]):
    #     y, x = coords[i]
    #     if (y, x) not in counted_point_set:
    #         for c in [(y, x-1), (y, x+1), (y-1, x), (y+1, x)]:
    #             if c in coords:


def conv2d(image, kernel):
    # Get image and kernel dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate output dimensions
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Initialize output
    output = np.zeros((output_height, output_width))
    
    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            output[i][j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output


def pattern_finding(f):
    with open(f, "rb") as f:
        data = pickle.load(f)
    gt_kps_3d = data["gt_p3d"]
    gt_p3d = np.stack(gt_kps_3d, axis=0)
    htree = create_human_tree("human3.6m")
    root = htree.root["index"]
    baseline_p3d = np.stack(data["baseline_p3d"], axis=0)
    gt_p2d = np.stack(data["gt_p2d"], axis=0)
    baseline_p2d = np.stack(data["ours_p2d"], axis=0).squeeze()
    baseline_conf = np.stack(data["baseline_confidences"], axis=0)
    ours_p3d = np.stack(data["ours_p3d"], axis=0)

    gt_p3d -= gt_p3d[:, root:root+1, :]
    ours_p3d -= ours_p3d[:, root:root+1, :]
    baseline_p3d -= baseline_p3d[:, root:root+1, :]
    mpjpe1 = np.mean(np.linalg.norm(gt_p3d - baseline_p3d, axis=-1))
    jpe2d1 = np.linalg.norm(gt_p2d - baseline_p2d, axis=-1)

    fig1 = plt.figure(figsize=(6, 4.5))
    ax1 = fig1.add_subplot()
    baseline_conf /= np.sum(baseline_conf, axis=1, keepdims=True)
    mask = np.logical_and(jpe2d1.flatten() > 20, baseline_conf.flatten() > 0.25)
    hist_limb = ax1.hist2d(baseline_conf.flatten()[mask], jpe2d1.flatten()[mask], bins=(50, 50))# , range=[[0, 0.6], [0, 25]])
    fig1.colorbar(hist_limb[-1], ax=ax1)
    ax1.set_xlabel("joint confidences")
    ax1.set_ylabel("joint orientation estimation error (per view)")
    plt.show()


def get_bl_tc():
    cfg = get_config("experiments/ResNet152/totalcapture-ResNet152-320x320-singlebranch.yaml")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_set = TotalCaptureMultiViewDataset(
        root_dir=cfg.DATASET.TC_ROOT,
        label_dir=cfg.DATASET.TC_LABELS,
        image_shape=tuple(cfg.MODEL.IMAGE_SIZE),
        is_train=False,
        transform=transform,
        crop=True,
        output_type=cfg.MODEL.REQUIRED_DATA,
        use_cameras=cfg.TEST.USE_CAMERAS,
        frame_sample_rate=1,
        sigma=cfg.MODEL.EXTRA.SIGMA
    )

    label = test_set.labels
    human_tree = create_human_tree('totalcapture')
    action_idx = np.array([l[0]['action'] for l in label])
    subject_idx = np.array([l[0]['subject'] for l in label])
    kps = [l[0]["joints_gt"] for l in label]
    kps = np.stack(kps, axis=0)
    for s in range(1, 6):
        mask = subject_idx == s
        # subjects
        bl_gt = np.mean(human_tree.get_bl_mat(kps[mask, ...]), axis=0)
        np.save(os.path.join('offlines', 'totalcapture', 'bone_lengths', f"S{s}_bl_gt.npy"), bl_gt)
    
    # model = MultiViewFPNet(cfg, False)
    # model.backbone.load_backbone_params(cfg.MODEL.BACKBONE_WEIGHTS, load_confidences=cfg.MODEL.USE_CONFIDENCE)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # with torch.no_grad():
    #     for i in Tposes:
    #         data = test_set.__getitem__(i)
    #         required_data = edict()
    #         for d_i, out in enumerate(cfg.MODEL.REQUIRED_DATA):
    #             if out in ["identity", "index", "subject"]:
    #                 # data_id
    #                 required_data[out] = data[d_i]
    #             else:
    #                 required_data[out] = data[d_i].to(device).float()
    #         out_values = model(required_data.images, required_data.projections, htree=human_tree, intrinsics=required_data.intrinsics,
    #                             rotation=required_data.rotation, cam_ctr=required_data.cam_ctr, fix_heatmap=cfg.MODEL.CO_FIXING.FIX_HEATMAP, 
    #                             pred_keypoints_3d=None, bone_lengths=None, sca_steps=cfg.TEST.SCA_STEPS)
    #         print(out_values)
 

def maha_test():
    # vec1 = torch.tensor([12, 33, 0], dtype=torch.float32)
    # vec2 = torch.tensor([44, 51, 1], dtype=torch.float32)
    # M = generate_Mahalanobis_mat(vec1, vec2)
    # print(M)
    # print(vec1.unsqueeze(0) @ M @ vec2.unsqueeze(1))
    batch_size, n_views, Nj = 8, 4, 17
    kps_2d = torch.randn(batch_size, n_views, Nj, 2, 1)
    proj_mats = torch.randn(batch_size, n_views, 3, 4)
    Ks = torch.randn(batch_size, n_views, 3, 3)
    htree = create_human_tree()
    mus = torch.randn(batch_size, n_views, Nj)*0.3 + 0.85
    kps_3d = optimize_wrt_param_mu(kps_2d, proj_mats, htree, "direct", {}, "op", mus=mus, intrinsics=Ks)
    print(kps_3d)


def vis_test():
    images = np.ones((4, 256, 256, 3), dtype=np.uint8) * 255
    vf1 = np.random.randn(4, 16, 3, 64, 64)
    vf2 = np.random.randn(4, 16, 3, 64, 64)
    fig = draw_di_vec_on_image(images, vf1, vf2)
    plt.show()
    # htree = create_human_tree()
    # offsetmap = np.ones((16, 3, 64, 64))
    # offsetmap[:, 2, :, :] = 0
    # offsetmap[:, 2, 40, 40] = 100
    # offsetmap1 = offsetmap.copy()
    # offsetmap1[:, 2, 41, 41] = -100
    # image = np.array(np.random.rand(256, 256, 3) * 10, dtype=np.uint8)
    # fig = vis_specific_vector_field(image, htree, offsetmap, offsetmap1)
    # plt.show()


def vec_err(vecs1, vecs2):
    prods = np.sum(vecs1 * vecs2, axis=-1) / (np.linalg.norm(vecs1, axis=-1) * np.linalg.norm(vecs2, axis=-1))
    prods[prods > 1] = 1
    prods[prods < -1] = -1
    # it doesn't matter it's positive or negtive
    return np.arccos(abs(prods))


def statistics_analysis():
    with open(os.path.join("debug", "h36m_analysis.pkl"), "rb") as pkf:
        data = edict(pickle.load(pkf))

    n_frames, n_views, n_joints, _, _ = data.pred_kps_2d.shape
    n_limbs = n_joints - 1
    data.confidences /= np.sum(data.confidences, axis=-1, keepdims=True)
    w_limb = data.confidences[:, :, :16]
    w_joint = data.confidences[:, :, 16:]
    homo_kps_3d = np.ones((n_frames, n_joints, 4))
    homo_kps_3d[:, :, :3] = data.pred_kps_3d
    homo_kps_2d = homo_kps_3d.reshape(n_frames, 1, n_joints, 4) @ data.projections.transpose(0, 1, 3, 2)
    gt_kps_2d = homo_kps_2d[:, :, :, :2] / homo_kps_2d[:, :, :, 2:3]
    err_joint_2d = np.linalg.norm(gt_kps_2d - data.pred_kps_2d.squeeze(-1), axis=-1)

    vecs1, vecs2 = data.pred_di_vecs.reshape(n_frames, n_views, n_limbs, 3), data.gt_di_vecs.reshape(n_frames, 1, n_limbs, 3)

    view_err_angles = vec_err(vecs1, vecs2)

    err_angle = vec_err(np.sum(vecs1 * w_limb.reshape(*w_limb.shape, 1), axis=1), vecs2.reshape(n_frames, n_limbs, 3))

    # Normalize
    # w_limb = w_limb / np.sum(w_limb, axis=-1, keepdims=True)
    # w_limb = np.log(w_limb)
    # view_err_angles = np.log(view_err_angles)
    plt.rc('font',family='Times') 
    font_times = fm.FontProperties(family='Times',size=12, stretch=0)
    fig1 = plt.figure(figsize=(6, 4.5))
    ax1 = fig1.add_subplot()
    hist_limb = ax1.hist2d(w_limb.flatten(), view_err_angles.flatten(), bins=(40, 40), range=[[0, 0.004], [0, 0.4]])
    fig1.colorbar(hist_limb[-1], ax=ax1)
    ax1.set_xlabel("limb confidences (x1.0e-3)")
    ax1.set_ylabel("limb orientation estimation error (per view)")

    fig2 = plt.figure(figsize=(6, 4.5))
    ax2 = fig2.add_subplot()
    hist_joint = ax2.hist2d(w_joint.flatten(), err_joint_2d.flatten(), bins=(40, 40), range=[[0, 0.1], [0, 6]])
    fig2.colorbar(hist_joint[-1], ax=ax2)
    ax2.set_xlabel("joint confidences")
    ax2.set_ylabel("joint orientation estimation error (per view)")

    fig3 = plt.figure(figsize=(6, 4.5))
    ax3 = fig3.add_subplot()
    hist_limb_all = ax3.hist2d(np.sum(w_limb, axis=1).flatten(), err_angle.flatten(), bins=(40, 40), range=[[0, 0.006], [0, 0.5]])
    fig3.colorbar(hist_joint[-1], ax=ax3)
    ax3.set_xlabel("joint confidences")
    ax3.set_ylabel("joint orientation estimation error (sum)")

    plt.show()

    # print(np.mean(view_err_angles))


def why_nan():
    with open(os.path.join('debug', 'error_inputs.pkl'), 'rb') as ipf:
        inputs = pickle.load(ipf)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = get_config("hourglass/hourglass2b-256x256")
    model = MultiViewFPNet(cfg, True)
    model.load_backbone_params(os.path.join('debug', 'error_state_dict.pth'), load_confidences=True)
    print("Pretrained 2D backbone loaded.")
    if torch.cuda.device_count() > 1 and cfg.TRAIN.BATCH_SIZE > 1 and cfg.TRAIN.DATA_PARALLEL:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model.to(device)

    human_tree = create_human_tree()
    out_values = model(inputs.images, inputs.projections, human_tree, inputs.intrinsics, rotation=inputs.rotation)
    print(out_values)


# class TestModule(nn.Module):
#     def __init__(self, bs, d):
#         super(TestModule, self).__init__()
#         self.svd_layer = rdsvd.apply
#         self.M = torch.randn(bs, d, d)
#         self.M = self.M + self.M.transpose(-1, -2)
#         self.M._requires_grad(True)

#     def forward(self, x):
#         x = x + self.M
#         u, s = self.svd_layer(x)
#         return u
    
#     def param_update(self, lr):
#         self.M -= self.M.grad  * lr


def svd_test():
    batch_size = 1
    d = 3
    n_its = 2000
    target = torch.randn(batch_size, d, d)
    target = target + target.transpose(-1, -2)
    torch.random.manual_seed(0)
    ut, s, u = torch.svd(target)
    loss_fn = nn.MSELoss()
    loss_ = []
    loss_rd = []
    M = torch.randn(batch_size, d, d, requires_grad=True)
    M_ = M.detach().clone().requires_grad_(True)
    lr = 0.0001
    for i in range(n_its):
        pred_u, pred_s = rdsvd.apply(M)
        # pred_ut, s, pred_u = torch.svd(M)
        loss = loss_fn(pred_u, u)
        loss.backward()
        M = (M - M.grad * lr).detach().requires_grad_(True)
        loss_rd.append(loss.item())
        if i > 2 and loss_rd[-1] > loss_rd[-2]:
            print(M, pred_s)
    
    for j in range(n_its):
        pred_u, pred_s, pred_v = torch.svd(M_)
        loss = loss_fn(pred_u, u)
        loss.backward()
        M_ = (M_ - M_.grad * lr).detach().requires_grad_(True)
        loss_.append(loss.item())

    
    plt.plot(loss_rd)
    plt.plot(loss_)
    plt.legend(["RDSVD", "SVD"])
    plt.show()


def recover_gaussian_kernels():
    # generate points
    h = w = 20
    md = 20
    heatmap = np.zeros((h, w))
    mask = np.ones((h, w))
    # np.random.seed(0)
    n_tops = 1
    selected_points = np.zeros((n_tops, 2))
    for i in range(n_tops):
        lam = np.random.rand() * 0.9 + 0.1
        coords = np.where(mask == 1)
        idx = np.random.randint(0, coords[0].shape[0])
        x, y = coords[1][idx], coords[0][idx] 
        x, y = 8, 12
        mask[max(y-md, 0): min(y+md, h), max(x-md, 0): min(x+md, w)] = 0
        # selected_points[i, :] = np.array([x, y])
        # generate heatmaps:
        dist = (np.arange(0, w).reshape(1, w) - x)**2 + (np.arange(0, h).reshape(h, 1) - y)**2
        heatmap += lam * np.exp(-dist / 10)

    sns.heatmap(heatmap)
    plt.show()


def generate_fusion_converter():
    element = bilinear_line_integral_offline((96, 96))
    # np.save(os.path.join('offlines', 'element_vmap.npy'), element)
    sns.heatmap(np.linalg.norm(element[17, 50], axis=0))
    plt.show()


def vis_cam_and_human_pose(htree, X_3d, Rs, Cs):
    """
    X_3d: 3D pose coordinates.
    Rs: rotation matrices. nv x 3 x 3
    Cs: camera centres. nv x 3
    """
    ## metric transform
    X_3d /= 1000
    Cs /= 1000
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    nv = Rs.shape[0]
    vec = Rs[:, 2, :]
    end = Cs + vec
    ax.quiver(Cs[:, 0], Cs[:, 1], Cs[:, 2], vec[:, 0], vec[:, 1], vec[:, 2], length=1)
    for v in range(nv):
        ax.text(Cs[v, 0], Cs[v, 1], Cs[v, 2], "%d"%(v+1))
    htree.draw_skeleton(ax, X_3d)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return fig


def gen_str(k, v):
    if type(v) == edict:
        kstr = '\n' + k + " = edict()\n"
        for kk, vv in v.items():
            kstr += gen_str(k + "." + kk, vv)
    elif type(v) == str:
        kstr = k + ' = \"' + v + '\"\n'
    else:
        kstr = k + ' = ' + str(v) + '\n'
    
    return kstr


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset1 = Human36MMonocularFeatureMapDataset(
        root_dir="/data1/share/dataset/human36m_multi-view/",
        label_dir="data/human36m-monocular-labels-GTbboxes.npy",
        sigma=2,
        image_shape=(256, 256),
        heatmap_shape=(64, 64),
        output_type=["images", "heatmap", "joint_vis"],
        is_train=True,
        transform=transform,
        crop=True,
    )

    dataset2 = MHADHeatmapDataset(
        root_dir="/data1/share/dataset/MHAD_Berkeley/stereo_camera",
        label_dir="/data1/share/dataset/MHAD_Berkeley/stereo_camera/extra/mhad-stereo-l-labels-GTbboxes.npy",
        image_shape=(256, 256),
        heatmap_shape=(64, 64),
        output_type=["images", "heatmap", "joint_vis"],
        transform=transform,
        test_sample_rate=1,
        is_train=True,
        rectificated=True,
        baseline='l',
        crop=True)
    
    js = JointDataset(dataset1, dataset2)

    test_loader = DataLoader(
        dataset=js,
        batch_size=64,
        shuffle=True,
        collate_fn = collate_pose,
        num_workers=1,
    )

    for i, d in enumerate(test_loader):
        for dt in d:
            print(dt.shape)
        if i > 2:
            break


# if __name__ == "__main__":
    # # case_study()
    # cfg = get_config("experiments/ResNet152/totalcapture-ResNet152-320x320-singlebranch.yaml")
    # print(gen_str("config", cfg))
    # get_bl_tc()
    # generate_fusion_converter()
    # pattern_finding(os.path.join("result_analysis", "h36m_pred_gt_noBad.pkl"))
    # recover_gaussian_kernels()
    # svd_test()
    # statistics_analysis()
    # vis_particular_frame()
