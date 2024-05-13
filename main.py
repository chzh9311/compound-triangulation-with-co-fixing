# ------------
# End-to-end training of the whole framework.
# ---------------
import os
import sys
import traceback
import shutil

import time
import logging
import yaml

import numpy as np
import pandas as pd
import pickle
from argparse import ArgumentParser

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from easydict import EasyDict as edict

from lib.dataset.joint import build_3D_dataset
from lib.models.field_pose_net import get_FPNet
from lib.models.MV_field_pose_net import MultiViewFPNet
from lib.utils.evaluate import soft_SMPJPE, MPJPE_abs, MPJPE_rel, MPLAE, MPLLE, vector_error, dire_map_pixelwise_loss, line_tri_loss, heatmap_central_loss
from lib.utils.DictTree import create_human_tree
from lib.utils.utils import make_logger, time_to_string
from lib.utils.functions import normalize, collate_pose
from config import update_config, save_config
from config import config as default_config
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

from lib.utils.vis import vis_2d_kps, vis_heatmap_data, vis_density, draw_di_vec_on_image, draw_heatmap_on_image
from lib.utils.vis import draw_vec_field_on_image, analyze_particular_frame, analyze_lof, vis_heatmap_and_gtpts, analyze_cofix, analyze_compound_tri
from tqdm import tqdm

# for VT testing
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

def train_one_epoch(config, epoch, train_loader, model, loss_fns, optimizer, human_tree, writer, logger, debug=False):
    model.train()
    size = len(train_loader.dataset)
    start = time.time()

    for batch_i, data in enumerate(train_loader):
        required_data = edict()
        model_out = edict()
        losses = edict()
        for d_i, out in enumerate(config.MODEL.REQUIRED_DATA):
            if out in ["identity", "index", "subject", "data_label"]:
                required_data[out] = data[d_i] 
            else:
                required_data[out] = data[d_i].to(device).float()
        # images, projections, gt_kps_3d, gt_dm, gt_mus, gt_bvs, intrinsics = [
        #     data[i].to(device).float() for i in range(len(data))]
        gt_kps_3d = required_data.keypoints3d
        required_data.num_joints = config.MODEL.NUM_JOINTS
        required_data.num_limbs = config.MODEL.NUM_LIMBS
        ## Fix_heatmap set to False by default.
        # with torch.autograd.set_detect_anomaly(True):
        out_values = model(htree=human_tree, **required_data)
        for i, k in enumerate(config.MODEL.MODEL_OUTPUT):
            model_out[k] = out_values[i]
        # heatmaps, pred_kps_2d, pred_kps_3d, pred_mus = model(images, projections, human_tree, intrinsics)
        # loss = torch.mean(loss_fn(pred_kps_3d, gt_kps_3d))
        losses["keypoints3d"] = loss_fns.coordinate(model_out.keypoints3d, required_data.keypoints3d, config.TRAIN.SOFT_EP)
        losses["heatmap"] = 4 * 4096 * loss_fns.heatmap(model_out.heatmap, required_data.keypoints2d_in_hm)
        if "lof" in model_out:
            gt_vecs = gt_kps_3d[:, human_tree.limb_pairs[:, 1], :] - gt_kps_3d[:, human_tree.limb_pairs[:, 0], :]
            local_gt_vecs = gt_vecs.reshape(gt_kps_3d.shape[0], 1, config.MODEL.NUM_LIMBS, 3) @ required_data.rotation.transpose(-1, -2)
            di_loss = loss_fns.di_map(local_gt_vecs[:, :, :, :config.MODEL.NUM_DIMS], model_out.lof, required_data.limb_vis)
            losses["lof"] = 1000 * di_loss
        if "keypoints3d_tri" in model_out:
            losses["keypoints3d_tri"] = loss_fns.coordinate(model_out.keypoints3d_tri, required_data.keypoints3d, config.TRAIN.SOFT_EP)
        if "lines_tri" in model_out:
            losses["lines_tri"] = loss_fns.line_tri_loss(*model_out.lines_tri, required_data.keypoints3d, human_tree.limb_pairs, config.TRAIN.SOFT_EP)

        # if "regularization" in model_out:
        #     reg_loss = torch.mean(model_out.regularization)
        #     loss = kp_loss # + di_loss * 500
        # else:
        #     loss = kp_loss
        loss = sum(losses.values())
        # loss = losses["keypoints3d"] + losses["lof"]
        losses["total"] = loss

        if torch.isnan(loss):
            stdt = model.state_dict()
            logger.error("The loss become Nan.")
            torch.save(stdt, os.path.join("debug", "error_state_dict.pth"))
            with open(os.path.join("debug", "error_inputs.pkl"), "wb") as stfile:
                pickle.dump(required_data, stfile)
            exit(-1)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_i % config.TRAIN.LOSS_FREQ == config.TRAIN.LOSS_FREQ - 1 or debug:
            current = batch_i * config.TRAIN.BATCH_SIZE
            log_info = f"Total loss: {losses.total:>5f}"
            for k in losses.keys():
                if k != 'total':
                    log_info += f", {k} loss: {losses[k]:>5f}"
            logger.info(log_info + f" currently {current:>7d}/{size:>7d}")
            # logger.info(f"Loss: {loss.item():>5f}, Kp loss: {kp_loss.item():>5f}, Di loss: {di_loss.item():>5f}, currently {current:>7d}/{size:>7d}")
            writer.add_scalars(
                "Training Loss",
                losses,
                global_step=epoch * size + batch_i * config.TRAIN.BATCH_SIZE
            )
        if batch_i % config.TRAIN.VIS_FREQ == config.TRAIN.VIS_FREQ - 1 or debug:
            vis_idx = np.random.randint(0, required_data.images.shape[0])
            images = required_data.images[vis_idx, ...].detach().cpu().numpy()
            images = np.stack([images[:, 2-i, ...] for i in range(3)], axis=3)
            images = np.round((images - images.min()) / (images.max() - images.min()) * 255).astype(np.uint8)
            P = required_data.projections.unsqueeze(2) # bs x nv x 1 x 3 x 4
            gt_homo_kp_2d = (P[..., :3] @ required_data.keypoints3d.view(P.shape[0], 1, -1, 3, 1) + P[..., 3:4]).squeeze(-1) # bs x nv x nj x 3
            gt_kps_2d = (gt_homo_kp_2d[..., :2] / gt_homo_kp_2d[..., 2:3]).detach().cpu().numpy()[vis_idx] # bs x nv x nj x 2
            pred_kps_2d = model_out.keypoints2d[vis_idx, ...].squeeze(-1).detach().cpu().numpy() # 4 x 17 x 2

            writer.add_figure(
                "Training vis - keypoints",
                vis_2d_kps(images, {"pred": pred_kps_2d, "gt": gt_kps_2d}, human_tree),
                global_step=epoch * size + batch_i * config.TRAIN.BATCH_SIZE
            )

            if "lof" in model_out:
                pred_lb_dm = model_out.lof[vis_idx, :, :config.MODEL.NUM_LIMBS, ...].detach().cpu().numpy()
                # writer.add_figure(
                #     "Training vis - vectors",
                #     draw_di_vec_on_image(images, pred_lb_dm),
                #     global_step=epoch * size + batch_i * config.TRAIN.BATCH_SIZE
                # )

                cam_idx = 0
                pred_lb_dm = np.linalg.norm(pred_lb_dm[cam_idx, ...], axis=1)
                pred_hm = model_out.heatmap #.view(required_data.images.shape[:2] + model_out.heatmap.shape[1:])
                pred_hm = pred_hm[vis_idx, cam_idx, :config.MODEL.NUM_JOINTS].detach().cpu().numpy()

                pred_hm = np.concatenate((pred_hm, pred_lb_dm), axis=0)
                writer.add_figure(
                    "Training vis - heatmaps",
                    vis_heatmap_data(images[cam_idx], pred_hm, None, config.MODEL.NUM_JOINTS, human_tree, "heatmap2d"),
                    global_step=epoch * size + batch_i * config.TRAIN.BATCH_SIZE
                )

            current_time = time.time() - start
            logger.info(f""" In this epoch:
            Time spent: {time_to_string(current_time)}
            Time per batch: {current_time / (batch_i+1):.2f}
            Time remaining: {time_to_string((len(train_loader) - batch_i - 1) / (batch_i + 1) * current_time)}
            """)

            if debug:
                return


def test_one_epoch(config, epoch, dataloader, model, test_loss_fns, human_tree, writer, logger, exp_name, bone_lengths=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    ndim = config.MODEL.NUM_DIMS
    criteria = ["MPJPE-ab", "MPJPE-re", "MPLAE", "MPLLE", "JPE2D", "XYError", "ZError", "n_samples"]
    if config.MODEL.CO_FIXING.FIX_HEATMAP:
        criteria += ["MPJPE-ab_combined", "MPJPE-re_combined", "MPLAE_combined", "MPLLE_combined"]
    if config.MODEL.USE_LOF:
        if "keypoints3d_tri" in config.MODEL.MODEL_OUTPUT:
            criteria += ["MPJPE-ab_tri", "MPJPE-re_tri", "MPLAE_tri", "MPLLE_tri"]
        if ndim == 3:
            criteria += ["LDE-single", "LDE w/o conf", "LDE w/ conf", "LDE2D", "weight ratio"]
        elif ndim == 2:
            criteria += ["LDE2D", "weight ratio"]
    result = {}
    for c in criteria:
        if c in ['n_samples']:
            result[c] = defaultdict(int)
        else:
            result[c] = defaultdict(list)
    pred_kps = {"keypoints3d":[], "index":[]}
    total_time = 0
    with torch.no_grad():
        for batch_i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            required_data = edict()
            for d_i, out in enumerate(config.MODEL.REQUIRED_DATA):
                if out in ["identity", "index", "subject"]:
                    # data_id
                    required_data[out] = data[d_i]
                else:
                    required_data[out] = data[d_i].to(device).float()
            gt_kps_3d = required_data.keypoints3d
            if bone_lengths is not None:
                if config.DATASET.NAME == "totalcapture":
                    bs = gt_kps_3d.shape[0]
                    bls = torch.tensor(bone_lengths[batch_i*bs:(batch_i+1)*bs], device=device).float()
                else:
                    bls = torch.tensor([bone_lengths[s] for s in required_data.subject], device=device).float()
            else:
                bls = None
            required_data.num_joints = config.MODEL.NUM_JOINTS
            required_data.num_limbs = config.MODEL.NUM_LIMBS
            out_values = model(htree=human_tree, **required_data)
                                           # out_values = model(required_data.images, required_data.projections, human_tree, required_data.intrinsics, required_data.rotation, required_data.cam_ctr,
                            #    fix_heatmap=config.MODEL.CO_FIXING.FIX_HEATMAP)
            # if batch_i == 99:
            #     print(total_time / 100)
            #     break
                
            model_out = edict()
            model_out_labels = config.MODEL.MODEL_OUTPUT + ["keypoints2d_combined", "keypoints3d_combined",
                                                            "lof_fixed", "heatmap_fixed"] * config.MODEL.CO_FIXING.FIX_HEATMAP
            for i, k in enumerate(model_out_labels):
                model_out[k] = out_values[i]

            P = required_data.projections.unsqueeze(2) # bs x nv x 1 x 3 x 4
            pred_kps_3d = model_out.keypoints3d

            pred_kps_2d = model_out.keypoints2d
                
            ## comment if no combined
            # debug_data["pred_kps_3d"].append(pred_kps_3d)
            # debug_data["gt_kps_3d"].append(gt_kps_3d)
            # line_loss = test_loss_fns.line_tri_loss(*model_out.lines_tri, required_data.keypoints3d, human_tree.limb_pairs, 1000)

            R = required_data.rotation.unsqueeze(2)
            gt_vecs = gt_kps_3d[:, human_tree.limb_pairs[:, 1], :] - gt_kps_3d[:, human_tree.limb_pairs[:, 0], :]
            homo_kp_2d = (P[..., :3] @ gt_kps_3d.view(P.shape[0], 1, -1, 3, 1) + P[..., 3:4]).squeeze(-1) # bs x nv x nj x 3
            gt_kps_2d = homo_kp_2d[..., :2] / homo_kp_2d[..., 2:3] # bs x nv x nj x 2
            for d_i in range(len(required_data.identity)):
                mpjpeab_original = test_loss_fns.coordinate[0](pred_kps_3d[d_i], gt_kps_3d[d_i]).item()
                mpjpere_original = test_loss_fns.coordinate[1](pred_kps_3d[d_i], gt_kps_3d[d_i], human_tree.root["index"]).item()
                result["MPJPE-ab"][required_data.identity[d_i]].append(mpjpeab_original)
                result["MPJPE-re"][required_data.identity[d_i]].append(mpjpere_original)
                result["MPLAE"][required_data.identity[d_i]].append(test_loss_fns.angle(pred_kps_3d[d_i], gt_kps_3d[d_i], human_tree.limb_pairs).item())
                result["MPLLE"][required_data.identity[d_i]].append(MPLLE(pred_kps_3d[d_i], gt_kps_3d[d_i], human_tree.limb_pairs).item())
                result["JPE2D"][required_data.identity[d_i]].append(torch.mean(torch.norm(pred_kps_2d[d_i].squeeze(-1) - gt_kps_2d[d_i], dim=-1)).item())
                gt_err = R[d_i, 0] @ (pred_kps_3d[d_i] - gt_kps_3d[d_i]).unsqueeze(-1)
                gt_err = gt_err.squeeze()
                result["XYError"][required_data.identity[d_i]].append(torch.mean(torch.norm(gt_err[..., :2], dim=-1)).item())
                result["ZError"][required_data.identity[d_i]].append(torch.mean(torch.abs(gt_err[..., 2])).item())
                result["n_samples"][required_data.identity[d_i]] += 1

                if config.MODEL.CO_FIXING.FIX_HEATMAP:
                    mpjpere_combined = test_loss_fns.coordinate[1](model_out.keypoints3d_combined[d_i], gt_kps_3d[d_i], human_tree.root["index"]).item()
                    mpjpeab_combined = test_loss_fns.coordinate[0](model_out.keypoints3d_combined[d_i], gt_kps_3d[d_i]).item()
                    result["MPJPE-re_combined"][required_data.identity[d_i]].append(mpjpere_combined)
                    result["MPJPE-ab_combined"][required_data.identity[d_i]].append(mpjpeab_combined)
                    result["MPLAE_combined"][required_data.identity[d_i]].append(test_loss_fns.angle(model_out.keypoints3d_combined[d_i], gt_kps_3d[d_i], human_tree.limb_pairs).item())
                    result["MPLLE_combined"][required_data.identity[d_i]].append(MPLLE(model_out.keypoints3d_combined[d_i], gt_kps_3d[d_i], human_tree.limb_pairs).item())
                    ## output
                    if abs(mpjpere_combined - mpjpere_original) > 0.02:
                        logger.debug(f"{batch_i*len(required_data.identity) + d_i}: original: {mpjpere_original:.2f}; combined: {mpjpere_combined:.2f}.")

                if config.MODEL.USE_LOF:
                    local_gtvecs = []
                    if "keypoints3d_tri" in config.MODEL.MODEL_OUTPUT:
                        mpjpeab_tri = test_loss_fns.coordinate[0](model_out.keypoints3d_tri[d_i], gt_kps_3d[d_i]).item()
                        mpjpere_tri = test_loss_fns.coordinate[1](model_out.keypoints3d_tri[d_i], gt_kps_3d[d_i], human_tree.root["index"]).item()
                        result["MPJPE-ab_tri"][required_data.identity[d_i]].append(mpjpeab_tri)
                        result["MPJPE-re_tri"][required_data.identity[d_i]].append(mpjpere_tri)
                        result["MPLAE_tri"][required_data.identity[d_i]].append(test_loss_fns.angle(model_out.keypoints3d_tri[d_i], gt_kps_3d[d_i], human_tree.limb_pairs).item())
                        result["MPLLE_tri"][required_data.identity[d_i]].append(MPLLE(model_out.keypoints3d_tri[d_i], gt_kps_3d[d_i], human_tree.limb_pairs).item())
                    if ndim == 3:
                        pred_gvec = []
                        n_views = model_out.di_vectors.shape[1]
                        for v in range(n_views):
                            pred_gvec.append(model_out.di_vectors[d_i, v, ...].reshape(config.MODEL.NUM_LIMBS, ndim) @ required_data.rotation[d_i, v, ...])
                            local_gtvecs.append(gt_vecs[d_i].reshape(config.MODEL.NUM_LIMBS, 3) @ required_data.rotation[d_i, v, ...].T) 
                        
                        pred_gvec = torch.stack(pred_gvec, dim=0)
                        local_gtvecs = torch.stack(local_gtvecs, dim=0)
                        lde_single = sum([vector_error(pred_gvec[i].reshape(-1, ndim), gt_vecs[d_i].view(-1, ndim)).item() for i in range(n_views)]) / n_views
                        # Limb Orientation Field
                        lde_woconf = vector_error(sum([pred_gvec[i].reshape(-1, ndim) for i in range(n_views)]), gt_vecs[d_i].view(-1, ndim)).item()
                        w_limbs = model_out.confidences[:, :, :config.MODEL.NUM_LIMBS]
                        w_pts = model_out.confidences[:, :, config.MODEL.NUM_LIMBS:]
                        lde_wconf = vector_error(sum([w_limbs[d_i, i].unsqueeze(-1) * pred_gvec[i].reshape(-1, ndim) for i in range(n_views)]), gt_vecs[d_i].view(-1, 3)).item()
                        result["LDE-single"][required_data.identity[d_i]].append(lde_single)
                        result["LDE w/o conf"][required_data.identity[d_i]].append(lde_woconf)
                        result["LDE w/ conf"][required_data.identity[d_i]].append(lde_wconf)
                        lde_2d = sum([vector_error(local_gtvecs[i].reshape(-1, 3)[:, :2], model_out.di_vectors[d_i, i].view(-1, ndim)[:, :2]).item() for i in range(n_views)]) / n_views
                    elif ndim == 2:
                        n_views = model_out.di_vectors.shape[1]
                        for v in range(n_views):
                            local_gtvecs.append(gt_vecs[d_i].reshape(config.MODEL.NUM_LIMBS, 3) @ required_data.rotation[d_i, v, ...].T)
                        local_gtvecs = torch.stack(local_gtvecs, dim=0)
                        w_limbs = model_out.confidences[:, :, :config.MODEL.NUM_LIMBS]
                        w_pts = model_out.confidences[:, :, config.MODEL.NUM_LIMBS:]
                        lde_2d = sum([vector_error(local_gtvecs[i].reshape(-1, 3)[:, :2], model_out.di_vectors[d_i, i].view(-1, ndim)).item() for i in range(n_views)]) / n_views
                    result["LDE2D"][required_data.identity[d_i]].append(lde_2d)
                    result["weight ratio"][required_data.identity[d_i]].append(torch.mean(w_limbs).item() / torch.mean(w_pts).item())

            if writer is not None:
                bs, nv = required_data.images.shape[:2]
                if config.TEST.VIS_FREQ > 0 and batch_i % (int(num_batches/config.TEST.VIS_FREQ)+1) == 0:
                    vis_idx = np.random.randint(0, required_data.images.shape[0])
                    # for vis_idx in range(bs):
                    images = required_data.images[vis_idx, ...].detach().cpu().numpy()
                    images = np.stack([images[:, 2-i, ...] for i in range(3)], axis=3)
                    images = np.round((images - images.min()) / (images.max() - images.min()) * 255).astype(np.uint8)
                    gt_kps_3d = required_data.keypoints3d[vis_idx, ...].detach().cpu().numpy() # 17 x 3
                    projection = required_data.projections[vis_idx, ...].detach().cpu().numpy() # 4 x 3 x 4
                    # di_vecs = model_out.di_vectors[vis_idx, ...].squeeze().detach().cpu().numpy()

                    pred_kps_2d = model_out.keypoints2d[vis_idx, ...].detach().cpu().numpy() # 4 x 17 x 2
                    gt_kps_2d = np.zeros((pred_kps_2d.shape[0], config.MODEL.NUM_JOINTS, 2))
                    for i in range(gt_kps_2d.shape[0]):
                        P = projection[i, ...]
                        homo_kp_3d = np.concatenate((gt_kps_3d.T, np.ones((1, gt_kps_3d.shape[0]))), axis=0)
                        homo_kp_2d = P @ homo_kp_3d
                        gt_kps_2d[i, ...] = (homo_kp_2d[:2, :] / homo_kp_2d[2:3, :]).T

                    if "lof" in model_out:
                        pred_lb_dm = model_out.lof[vis_idx, :, :config.MODEL.NUM_LIMBS, ...].detach().cpu().numpy()
                        cam_idx = np.random.randint(0, nv)
                        pred_lb_dm_cam = np.linalg.norm(pred_lb_dm[cam_idx, ...], axis=1)
                        pred_hm = model_out.heatmap #.view(required_data.images.shape[:2] + model_out.heatmap.shape[1:])
                        pred_hm = pred_hm[vis_idx, cam_idx, :config.MODEL.NUM_JOINTS].detach().cpu().numpy()

                        pred_hm = np.concatenate((pred_hm, pred_lb_dm_cam), axis=0)
                        writer.add_figure(
                            f"Training vis - heatmaps cam {cam_idx}",
                            vis_heatmap_data(images[cam_idx], pred_hm, None, config.MODEL.NUM_JOINTS, human_tree, "heatmap2d"),
                            global_step = epoch * size + batch_i * bs + vis_idx
                        )

                    if config.MODEL.CO_FIXING.FIX_HEATMAP:
                        # fixed_kps_2d = model_out.keypoints2d_fixed.squeeze(-1)[vis_idx, ...].detach().cpu().numpy()
                        combined_kps_2d = model_out.keypoints2d_combined.squeeze(-1)[vis_idx, ...].detach().cpu().numpy()
                        pt_dict = OrderedDict({"pred": pred_kps_2d, "combined": combined_kps_2d, "gt": gt_kps_2d})
                    else:
                        pt_dict = OrderedDict({"pred": pred_kps_2d, "gt": gt_kps_2d})
                    writer.add_figure(
                        "Test vis - keypoints",
                        vis_2d_kps(images, pt_dict, human_tree),
                        epoch * size + batch_i * bs + vis_idx
                        )

                    # writer.add_figure(
                    #     "lof",
                    #     vis_heatmap_data(images[0], pred_hm, gt_hm, 17, human_tree, "heatmap2d", pred_limb_labels=pred_limb_labels, gt_limb_labels=gt_limb_labels),
                    #     global_step=epoch * size + batch_i * config.TRAIN.BATCH_SIZE
                    # )
    data_ids = list(result["MPJPE-ab"].keys())
    result_df = pd.DataFrame(None, index=criteria, columns=data_ids)
    avg_result = {}
    for c in criteria:
        if c not in ['n_samples']:
            err_list = []
            for data_id in data_ids:
                err_list += result[c][data_id]
                result_df.loc[c, data_id] = sum(result[c][data_id]) / len(result[c][data_id])
            avg_result[c] = result_df.loc[c, "mean"] = sum(err_list) / len(err_list)
        else:
            for data_id in data_ids:
                result_df.loc[c, data_id] = result[c][data_id]
            avg_result[c] = result_df.loc[c, "mean"] = sum(result[c].values()) / len(result[c])
    
    log_info = "\n".join([f"{c}: {avg_result[c]}" for c in criteria])

    logger.info(log_info)
    if writer is not None:
        writer.add_scalars("Testing criteria", avg_result, epoch)
        result_df.to_csv(os.path.join("log", "end2end", exp_name, f"result_epoch{epoch}.csv"))
    if 'MPJPE-re_combined' in avg_result:
        return avg_result["MPJPE-re_combined"], result_df
    else:
        return avg_result["MPJPE-re"], result_df


def run_model(cfg, runMode='test', debug=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_set = build_3D_dataset(cfg, transform, True, True)
    test_set = build_3D_dataset(cfg, transform, False, True)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        collate_fn=collate_pose,
        num_workers=cfg.TRAIN.NUM_WORKERS
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=cfg.TEST.SHUFFLE,
        collate_fn=collate_pose,
        num_workers=cfg.TEST.NUM_WORKERS
    )
    htree = create_human_tree(cfg.DATASET.NAME)
    print("Finished loading data.")
    is_train = runMode == "train"

    if cfg.DATASET.NAME == "joint":
        cfg.MODEL.NUM_JOINTS = cfg.MODEL.NUM_JOINTS1 + cfg.MODEL.NUM_JOINTS2
        cfg.MODEL.NUM_LIMBS = cfg.MODEL.NUM_LIMBS1 + cfg.MODEL.NUM_LIMBS2
        cfg.MODEL.REQUIRED_DATA.append("data_label")
        model = MultiViewFPNet(cfg, is_train)
        cfg.MODEL.NUM_JOINTS = cfg.MODEL.NUM_JOINTS1
        cfg.MODEL.NUM_LIMBS = cfg.MODEL.NUM_LIMBS1
    else:
        model = MultiViewFPNet(cfg, is_train)

    if runMode == "train":
        model.backbone.load_backbone_params(cfg.MODEL.PRETRAINED)
        print(f"Pretrained 2D backbone loaded from {cfg.MODEL.PRETRAINED}.")
    else:
        model.backbone.load_backbone_params(cfg.MODEL.BACKBONE_WEIGHTS)
        print(f"2D backbone weights loaded from {cfg.MODEL.BACKBONE_WEIGHTS}.")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)

    if len(cfg.GPUS) > 1 and cfg.TRAIN.BATCH_SIZE > 1 and cfg.TRAIN.DATA_PARALLEL:
        print(f"Using GPU devices {cfg.GPUS}.")
        model = nn.DataParallel(model, device_ids=cfg.GPUS)
    model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    exp_name = f"{runMode}_{time.strftime('%Y%m%d_%H%M%S')}_{cfg.DATASET.NAME}_{cfg.MODEL.BACKBONE}"
    exp_dir = os.path.join("log", "end2end", exp_name)

    train_loss_fns = edict({"coordinate": soft_SMPJPE, "di_map": dire_map_pixelwise_loss, "line_tri_loss": line_tri_loss, "heatmap": heatmap_central_loss})
    test_loss_fns = edict({"coordinate": (MPJPE_abs, MPJPE_rel), "scalar": nn.L1Loss(), "angle": MPLAE})

    try:
        if cfg.TEST.WRITE_LOG:
            writer = SummaryWriter(os.path.join(exp_dir, "tb"))
            logger = make_logger(exp_name, os.path.join(exp_dir, "experiment.log"), logging.INFO)
            out_path = os.path.join(exp_dir, "weights")
            save_config(cfg, os.path.join(exp_dir, "config.yaml"))
        else:
            writer = None
            logger = make_logger(exp_name, None, level=logging.INFO)
        
        if runMode == "train":
            if cfg.TRAIN.CONTINUE:
                with open(cfg.TRAIN.CHECKPOINT, "rb") as ckpf:
                    prev_ckeckpoint = pickle.load(ckpf) #, map_location={"cuda:2":"cuda:0", "cuda:3":"cuda:1"})
                model.load_state_dict(prev_ckeckpoint["model"], strict=True)
                best_loss = prev_ckeckpoint["loss"]
                optimizer.load_state_dict(prev_ckeckpoint["optimizer"])
                start_epoch = prev_ckeckpoint["epoch"] + 1
                logger.info(f"Loaded checkpoint from {cfg.TRAIN.CHECKPOINT}")
            else:
                # Initialize
                start_epoch = 0
                best_loss = 60

            for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):
                logger.info(f"{20*'-'} Starting the {epoch}th epoch {20*'-'}")
                logger.info("Start Training...")
                train_one_epoch(cfg, epoch, train_loader, model, train_loss_fns, optimizer, htree, writer, logger, debug)

                logger.info("Start Testing...")
                current_loss, _ = test_one_epoch(cfg, epoch, test_loader, model, test_loss_fns, htree, writer, logger, exp_name)

                # Dump checkpoint
                state = {}
                state["optimizer"] = optimizer.state_dict()
                state["model"] = model.state_dict()
                state["epoch"] = epoch
                state["loss"] = current_loss
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                with open (os.path.join(out_path, f"checkpoint_epoch{epoch}_loss{current_loss:4e}.pkl"), "wb") as ckpt:
                    pickle.dump(state, ckpt)
                logger.info(f"Dumped Checkpoint to {out_path}")

                if current_loss < best_loss:
                    best_loss = current_loss
                    torch.save(model.state_dict(), os.path.join(out_path, "best.pth"))
                    logger.info(f"saved weights to {out_path}")
        else:
            loss, result = test_one_epoch(cfg, 0, test_loader, model, test_loss_fns, htree, writer, logger, exp_name, None)

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if os.path.exists(exp_dir) and debug:
            shutil.rmtree(exp_dir)
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=sys.stdout)
        exit(-1)


def main():
    # cfg = get_config("experiments/ResNet152/totalcapture-ResNet152-320x320-singlebranch.yaml")
    # cfg = get_config("experiments/ResNet152/totalcapture-ResNet152-320x320.yaml")
    # cfg = get_config("experiments/ResNet152/totalcapture-ResNet152-320x320-volumetric.yaml")
    # cfg = get_config("experiments/ResNet152/human3.6m-ResNet152-384x384-volumetric.yaml")
    # cfg = get_config("experiments/ResNet152/human3.6m-ResNet152-384x384.yaml")
    # cfg = get_config("experiments/ResNet152/human3.6m-ResNet152-384x384-singlebranch.yaml")
    # cfg = get_config("openpose/h36m-openpose-backbone-384x384")
    # cfg = get_config("hourglass/hourglass2b-256x256")
    parser = ArgumentParser(
        prog="End-to-end Pretraining",
        description="network training & testing program."
    )
    parser.add_argument("--cfg", default="experiments/ResNet152/totalcapture-ResNet152-320x320-singlebranch.yaml")
    parser.add_argument("--runMode")
    parser.add_argument("-e", "--epochs")
    parser.add_argument("-d", "--dir", help="the log directory of pretrained backbone to be tested.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dump", default=None)
    args = parser.parse_args()
    cfg = update_config(default_config, args)
    run_model(cfg, args.runMode, debug=args.debug)


if __name__ == "__main__":
    main()
    # test_2views()
    # cfg1 = get_config("experiments/ResNet152/human3.6m-ResNet152-384x384-singlebranch.yaml")
    # cfg2 = get_config("experiments/ResNet152/human3.6m-ResNet152-384x384.yaml")

    # cfg1 = get_config("experiments/ResNet152/totalcapture-ResNet152-320x320-singlebranch.yaml")
    # cfg2 = get_config("experiments/ResNet152/totalcapture-ResNet152-320x320.yaml")
    # qualititative_analysis(cfg1, cfg2)
