# ------------
# Pretraining of the backbones
# ---------------
import os
import time

import numpy as np
import pickle
from random import randrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Visualize
from matplotlib import pyplot as plt
import seaborn as sns

import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import logging

from easydict import EasyDict as edict
from collections import OrderedDict
from argparse import ArgumentParser

from lib.dataset.joint import build_2D_dataset
from lib.models.field_pose_net import get_FPNet
# from lib.models.ditehrnet import DiteHRNet
from lib.utils.DictTree import create_human_tree
from lib.utils.functions import fit_1d_density, calc_vanish, calc_vanish_from_vmap, normalize, collate_pose
from lib.utils.evaluate import heatmap_MSE, heatmap_weighted_MSE, vector_error, heatmap_norm_max_dist, dire_map_error, dire_map_angle_err
from lib.utils.utils import make_logger, time_to_string
from lib.utils.vis import vis_heatmap_data, vis_specific_vector_field
from config import get_config, save_config, update_config
from config import config as default_cfg

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(epoch, config, dataloader, model, loss_fns, optimizer, htree, writer, logger, debug=False):
    model.train()
    size = len(dataloader.dataset)
    start = time.time()
    use_lof = config.MODEL.USE_LOF
    required = config.MODEL.REQUIRED_DATA.copy()
    out_labels = config.MODEL.BACKBONE_OUTPUT

    for batch_i, data in enumerate(dataloader):
        required_data = edict()
        model_out = edict()
        for d_i, out in enumerate(required):
            required_data[out] = data[d_i].to(device).float()

        out_values = model(required_data.images)
        preds = edict()
        losses = edict()
        for i, k in enumerate(out_labels):
            model_out[k] = out_values[i] if len(out_labels) > 1 else out_values
        if config.DATASET.NAME == "joint":
            data_ids = required_data.data_label.view(-1, 1, 1, 1)
            model_out['heatmap'] = data_ids * model_out['heatmap'][:, :config.MODEL.NUM_JOINTS1]\
                                   + (1 - data_ids) * model_out['heatmap'][:, config.MODEL.NUM_JOINTS1:]
            if 'lof' in out_labels:
                model_out['lof'] = data_ids * model_out['lof'][:, :3 * config.MODEL.NUM_LIMBS1]\
                                   + (1 - data_ids) * model_out['lof'][:, 3 * config.MODEL.NUM_LIMBS1:]
        for out in model_out:
            preds[out] = model_out[out]
            if out == 'heatmap':
                losses[out] = loss_fns[out](preds[out], required_data[out], 'joint', required_data['joint_vis'])
            elif out == 'lof':
                losses[out] = loss_fns['lof'](preds[out], required_data[out], 'limb', required_data['limb_vis'])

        loss = sum([v for v in losses.values()])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # Handle with gradient explosion
        # if reg_method == "heatmap2d":
        #     nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
        # visualization with tensorboard
        if batch_i % config.TRAIN.LOSS_FREQ == config.TRAIN.LOSS_FREQ - 1 or debug:
            current = batch_i * config.TRAIN.BATCH_SIZE
            loss_info = f"Total loss: {loss.item():.4e}"
            for out in out_labels:
                if out != "stacked":
                    loss_info += f", {out} loss: {losses[out].item():.4e}"
            loss_info += f", currently {current:>7d}/{size:>7d}."
            logger.info(loss_info)
            loss_values = {"Total loss": loss.item()}
            for k in losses.keys():
                loss_values[k+" loss"] = losses[k].item()
            writer.add_scalars(
                "training loss", loss_values, epoch * size + batch_i*config.TRAIN.BATCH_SIZE
            )

        if batch_i % config.TRAIN.VIS_FREQ == config.TRAIN.VIS_FREQ - 1 or debug:
            vis_idx = np.random.randint(0, required_data.images.shape[0])
            images = required_data.images[vis_idx, ...].detach().cpu().numpy() # type: ignore
            image = np.stack([images[i, ...] for i in range(images.shape[0])], axis=2)
            image = np.round((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

            pred_hm = model_out.heatmap[vis_idx, ...].detach().cpu().numpy()
            gt_hm = required_data.heatmap[vis_idx, ...].detach().cpu().numpy()
            Nj = pred_hm.shape[0]

            if "lof" in out_labels:
                bs, _, h, w = required_data.lof.shape
                ndim = config.MODEL.NUM_DIMS
                pred_lb_dm = model_out.lof[vis_idx, ...].view(-1, ndim, h, w).detach().cpu().numpy()
                pred_limb_labels = normalize(np.sum(pred_lb_dm, axis=(2, 3)), dim=1, tensor=False)
                pred_lb_dm = np.linalg.norm(pred_lb_dm, axis=1)
                gt_lb_dm = required_data.lof[vis_idx, ...].view(-1, ndim, h, w).detach().cpu().numpy()
                gt_limb_labels = normalize(np.sum(gt_lb_dm, axis=(2, 3)), dim=1, tensor=False)
                gt_lb_dm = np.linalg.norm(gt_lb_dm, axis=1)
                pred_hm = np.concatenate((pred_hm, pred_lb_dm), axis=0)
                gt_hm = np.concatenate((gt_hm, gt_lb_dm), axis=0)

                writer.add_figure(
                    "training vis",
                    vis_heatmap_data(image, pred_hm, gt_hm, Nj, htree, use_lof, pred_limb_labels=pred_limb_labels, gt_limb_labels=gt_limb_labels),
                    global_step=epoch * size + batch_i * config.TRAIN.BATCH_SIZE
                )
            else:
                writer.add_figure(
                    "training vis",
                    vis_heatmap_data(image, pred_hm, gt_hm, Nj, htree, use_lof),
                    global_step=epoch * size + batch_i * config.TRAIN.BATCH_SIZE
                )

            current_time = time.time() - start
            logger.info(f""" In this epoch:
            Time spent: {time_to_string(current_time)}
            Time per batch: {current_time / (batch_i+1):.2f}
            Time remaining: {time_to_string((len(dataloader) - batch_i - 1) / (batch_i + 1) * current_time)}
            """)
            if debug:
                break


def test_one_epoch(epoch, config, dataloader, model, loss_fns, htree, writer, logger):
    size = len(dataloader.dataset)
    start = time.time()
    use_lof = config.MODEL.USE_LOF
    required = config.MODEL.REQUIRED_DATA
    out_labels = config.MODEL.BACKBONE_OUTPUT
    model.eval()
    losses = edict()
    for out in out_labels:
        losses[out] = 0
    with torch.no_grad():
        for batch_i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            required_data = edict()
            model_out = edict()
            for d_i, out in enumerate(required):
                required_data[out] = data[d_i].to(device).float()

            out_values = model(required_data.images)
            preds = edict()
            for i, k in enumerate(out_labels):
                model_out[k] = out_values[i] if len(out_labels) > 1 else out_values
            if config.DATASET.NAME == "joint":
                data_ids = required_data.data_label.view(-1, 1, 1, 1)
                model_out['heatmap'] = data_ids * model_out['heatmap'][:, :config.MODEL.NUM_JOINTS1]\
                                       + (1 - data_ids) * model_out['heatmap'][:, config.MODEL.NUM_JOINTS1:]
                if 'lof' in out_labels:
                    model_out['lof'] = data_ids * model_out['lof'][:, :3 * config.MODEL.NUM_LIMBS1]\
                                       + (1 - data_ids) * model_out['lof'][:, 3 * config.MODEL.NUM_LIMBS1:]
            for out in model_out:
                preds[out] = model_out[out]
            for k in losses.keys():
                if k == "heatmap":
                    losses[k] += loss_fns[k](preds[k][:, :cfg.MODEL.NUM_JOINTS, ...], required_data[k], required_data.joint_vis).item() * preds[k].shape[0]
                elif k == "lof":
                    bs, _, h, w = required_data.lof.shape
                    ndim = config.MODEL.NUM_DIMS
                    required_data.lof = required_data.lof.view(bs, -1, ndim, h, w)
                    losses[k] += loss_fns[k](preds[k][:, :cfg.MODEL.NUM_LIMBS*ndim, ...].view(*required_data.lof.shape), required_data[k], required_data.limb_vis).item() * preds[k].shape[0]
                    if not (losses[k] > 0):
                        print("nan")

            if batch_i % (int(len(dataloader)/config.TEST.VIS_FREQ)+1) == 0 and writer is not None:
                vis_idx = np.random.randint(0, required_data.images.shape[0])
                images = required_data.images[vis_idx, ...].detach().cpu().numpy()
                image = np.stack([images[i, ...] for i in range(images.shape[0])], axis=2)
                image = np.round((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

                pred_hm = model_out.heatmap.squeeze()[vis_idx, ...].detach().cpu().numpy()
                gt_hm = required_data.heatmap[vis_idx, ...].detach().cpu().numpy()
                Nj = pred_hm.shape[0]

                if "lof" in out_labels:
                    bs, Nb, _, h, w = required_data.lof.shape
                    pred_lb_dm = model_out.lof[vis_idx, ...].view(-1, ndim, h, w).detach().cpu().numpy()
                    pred_limb_labels = normalize(np.sum(pred_lb_dm, axis=(2, 3)), dim=1, tensor=False)
                    pred_lb_dm = np.linalg.norm(pred_lb_dm, axis=1)
                    gt_lb_dm = required_data.lof[vis_idx, ...].view(-1, ndim, h, w).detach().cpu().numpy()
                    gt_limb_labels = normalize(np.sum(gt_lb_dm, axis=(2, 3)), dim=1, tensor=False)
                    gt_lb_dm = np.linalg.norm(gt_lb_dm, axis=1)
                    pred_hm = np.concatenate((pred_hm, pred_lb_dm), axis=0)
                    gt_hm = np.concatenate((gt_hm, gt_lb_dm), axis=0)

                    writer.add_figure(
                        "Testing vis",
                        vis_heatmap_data(image, pred_hm, gt_hm, Nj, htree, use_lof, pred_limb_labels=pred_limb_labels, gt_limb_labels=gt_limb_labels),
                        global_step=epoch * size + batch_i * config.TRAIN.BATCH_SIZE
                    )

                writer.add_figure(
                    "Testing vis",
                    vis_heatmap_data(image, pred_hm, gt_hm, Nj, htree, use_lof),
                    global_step=epoch * size + batch_i * config.TRAIN.BATCH_SIZE
                )

    for k in losses.keys():
        losses[k] /= size

    losses.total = sum([v for v in losses.values()])
    if writer is not None:
        writer.add_scalars(
            "Testing loss", losses, epoch)
    log_info = f"Total loss: {losses.total:.4e}"
    for k in losses.keys():
        if k != 'total':
            log_info += f", {k} loss: {losses[k]:.4e}"
    logger.info(log_info)

    return losses.total


def train(cfg, debug=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_set = build_2D_dataset(cfg, transform, True, True)
    test_set = build_2D_dataset(cfg, transform, False, True)

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

    exp_name = f"Training_{time.strftime('%Y%m%d_%H%M%S')}_{cfg.DATASET.NAME}_{cfg.MODEL.BACKBONE}"
    exp_path = os.path.join("./log", "backbone", exp_name)
    os.mkdir(exp_path)
    logger = make_logger(name=exp_name, filename=os.path.join(exp_path, "experiment.log"), level=logging.INFO)
    if cfg.DATASET.NAME == "joint":
        cfg.MODEL.NUM_JOINTS = cfg.MODEL.NUM_JOINTS1 + cfg.MODEL.NUM_JOINTS2
        cfg.MODEL.NUM_LIMBS = cfg.MODEL.NUM_LIMBS1 + cfg.MODEL.NUM_LIMBS2
        cfg.MODEL.REQUIRED_DATA.append("data_label")
    model = get_FPNet(cfg, is_train=True, pretrain=True)

    logger.info("Finished loading data.")
    logger.info(f"Using {device} device.")

    if len(cfg.GPUS) > 1 and cfg.TRAIN.DATA_PARALLEL:
        logger.info(f"Using {len(cfg.GPUS)} GPUs.")
        model = nn.DataParallel(model, cfg.GPUS)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    htree = create_human_tree(cfg.DATASET.NAME)

    # Record information about the experiment.
    out_path = os.path.join(exp_path, "weights")
    save_config(cfg, os.path.join(exp_path, "config.yaml"))
    htree.save(os.path.join(exp_path, "tree.json"))
    writer = SummaryWriter(os.path.join(exp_path, "tb"))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.SCHEDULER.STEP, cfg.TRAIN.SCHEDULER.GAMMA)

    if cfg.TRAIN.CONTINUE:
        with open(cfg.TRAIN.CHECKPOINT, "rb") as ckpf:
            prev_ckeckpoint = pickle.load(ckpf)
        model.load_state_dict(prev_ckeckpoint["model"], strict=True)
        best_loss = prev_ckeckpoint["loss"]
        optimizer.load_state_dict(prev_ckeckpoint["optimizer"])
        if cfg.TRAIN.MID_CHECKPOINTS == 0:
            start_epoch = prev_ckeckpoint["epoch"] + 1
            # for i in range(start_epoch):
            #     scheduler.step()
            start_sub_epoch = 0
            seed = int(time.time())
        if cfg.TRAIN.MID_CHECKPOINTS > 0:
            seed = prev_ckeckpoint["random_seed"]
            start_epoch = prev_ckeckpoint["epoch"]
            start_sub_epoch = prev_ckeckpoint["sub_epoch"] + 1
            if start_sub_epoch == cfg.TRAIN.MID_CHECKPOINTS:
                start_epoch += 1
                start_sub_epoch = 0
        logger.info(f"Loaded checkpoint from {cfg.TRAIN.CHECKPOINT}")
    else:
        # Initialize
        start_epoch = 0
        start_sub_epoch = 0
        seed = int(time.time())
        best_loss = 1

    # Start epochs
    l1loss = nn.L1Loss(reduction='mean')
    train_loss_fns = edict({"heatmap": heatmap_weighted_MSE, "lof": heatmap_weighted_MSE})
    test_loss_fns = edict({"heatmap": heatmap_norm_max_dist, "mu": l1loss, "vector": vector_error, "lof": dire_map_angle_err})
    dnum = len(train_set)

    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):
        logger.info(f"{20*'-'} Starting the {epoch}th epoch {20*'-'}")
        # Train for one epoch
        logger.info("Start Training...")
        if cfg.TRAIN.MID_CHECKPOINTS == 0:
            train_one_epoch(epoch, cfg, train_loader, model, train_loss_fns, optimizer, htree, writer, logger, debug)

            # Test for the current epoch
            logger.info("Start Testing...")
            current_loss = test_one_epoch(epoch, cfg, test_loader, model, test_loss_fns, htree, writer, logger)

            # Dump checkpoint
            state = {}
            state["optimizer"] = optimizer.state_dict()
            state["model"] = model.state_dict()
            state["epoch"] = epoch
            state["loss"] = current_loss
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            with open(os.path.join(out_path, f"checkpoint_epoch{epoch}_loss{current_loss:4e}.pkl"), "wb") as ckpt:
                pickle.dump(state, ckpt)
            logger.info(f"Dumped Checkpoint to {out_path}")

            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(model.state_dict(), os.path.join(out_path, f"best.pth"))
                logger.info(f"saved weights to {out_path}")
        else:
            nparts = cfg.TRAIN.MID_CHECKPOINTS + 1

            # use previous seed only in the first iteration.
            if cfg.TRAIN.CONTINUE and start_sub_epoch:
                pass
            else:
                seed = int(time.time())
            cfg.TRAIN.CONTINUE = False
            np.random.seed(seed)

            it_order = np.arange(dnum)
            if cfg.TRAIN.SHUFFLE:
                np.random.shuffle(it_order)
            for m in range(start_sub_epoch, nparts):
                train_set.set_sample_order(it_order[int(m*dnum/nparts):int((m+1)*dnum/nparts)])
                mid_train_loader = DataLoader(
                    dataset=train_set,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    shuffle=cfg.TRAIN.SHUFFLE,
                    collate_fn=collate_pose,
                    num_workers=cfg.TRAIN.NUM_WORKERS
                )

                train_one_epoch(epoch*nparts+m+1, cfg, mid_train_loader, model, train_loss_fns, optimizer, htree, writer, logger, debug)

                # Test for the current epoch
                logger.info("Start Testing...")
                current_loss = test_one_epoch(epoch*nparts+m+1, cfg, test_loader, model, test_loss_fns, htree, writer, logger)

                # Dump checkpoint
                state = {}
                state["optimizer"] = optimizer.state_dict()
                state["model"] = model.state_dict()
                state["epoch"] = epoch
                state["loss"] = current_loss
                state["random_seed"] = seed
                state["sub_epoch"] = m
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                with open (os.path.join(out_path, f"checkpoint_epoch{epoch + (m+1)/nparts:.3f}_loss{current_loss:4e}.pkl"), "wb") as ckpt:
                    pickle.dump(state, ckpt)
                logger.info(f"Dumped Checkpoint to {out_path}")

                if current_loss < best_loss:
                    best_loss = current_loss
                    torch.save(model.state_dict(), os.path.join(out_path, f"best.pth"))
                    logger.info(f"saved weights to {out_path}")
            start_sub_epoch = 0
        scheduler.step()
    writer.close()


def test(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_set = build_2D_dataset(cfg, transform, False, True)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=cfg.TEST.SHUFFLE,
        collate_fn=collate_pose,
        num_workers=cfg.TEST.NUM_WORKERS
    )

    exp_name = f"Testing_{time.strftime('%Y%m%d_%H%M%S')}_{cfg.DATASET.NAME}_{cfg.MODEL.BACKBONE}"
    exp_path = os.path.join("./log", "backbone", exp_name)
    os.mkdir(exp_path)
    logger = make_logger(exp_name, os.path.join(exp_path, "experiment.log"), logging.INFO)

    logger.info("Finished loading data.")
    logger.info(f"Using {device} device.")
    if cfg.DATASET.NAME == "joint":
        cfg.MODEL.NUM_JOINTS = cfg.MODEL.NUM_JOINTS1 + cfg.MODEL.NUM_JOINTS2
        cfg.MODEL.NUM_LIMBS = cfg.MODEL.NUM_LIMBS1 + cfg.MODEL.NUM_LIMBS2
        cfg.MODEL.REQUIRED_DATA.append("data_label")
    model = get_FPNet(cfg, is_train=False, pretrain=True)

    if len(cfg.GPUS) > 1 and cfg.TEST.DATA_PARALLEL:
        logger.info(f"Using {len(cfg.GPUS)} GPUs.")
        model = nn.DataParallel(model, cfg.GPUS)
    model.to(device)

    htree = create_human_tree()
    with open(cfg.MODEL.BACKBONE_WEIGHTS, "rb") as wf:
        if cfg.MODEL.BACKBONE_WEIGHTS[-4:] == ".pkl":
            checkpoint = pickle.load(wf)
            state_dict = checkpoint["model"]
        else:
            state_dict = torch.load(wf)
        goal = model.state_dict()
        for k in state_dict.keys():
            if k.startswith('module.final_layer'):
                state_dict[k] = state_dict[k][:goal[k].shape[0], ...]
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded trained weights from {cfg.MODEL.BACKBONE_WEIGHTS}.")

    # Record information about the experiment.
    save_config(cfg, os.path.join(exp_path, "config.yaml"))
    htree.save(os.path.join(exp_path, "tree.json"))
    if cfg.TEST.WRITE_LOG:
        writer = SummaryWriter(os.path.join(exp_path, "tb"))
    else:
        writer = None

    # Loss function
    l1loss = nn.L1Loss(reduction='mean')
    test_loss_fns = edict({"heatmap": heatmap_norm_max_dist, "scalar": l1loss, "vector": vector_error, "lof": dire_map_angle_err})

    logger.info("Start testing...")
    current_loss = test_one_epoch(0, cfg, test_loader, model, test_loss_fns, htree, writer, logger)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    # cfg = get_config('ResNet152/totalcapture-ResNet152-320x320-singlebranch-backbone')
    # cfg = get_config('experiments/ResNet152/totalcapture-ResNet152-320x320-singlebranch-backbone.yaml')
    # cfg = get_config("ResNet50/human3.6m-ResNet50-256x256-backbone")
    # cfg = get_config("ResNet152/human3.6m-ResNet152-384x384-backbone")
    # cfg = get_config("openpose/h36m-openpose-backbone-384x384")
    # cfg = get_config("hourglass/hourglass2b-backbone")
    parser = ArgumentParser(
        prog="Backbone Pretraining",
        description="Backbone training & testing program."
    )
    parser.add_argument("--cfg")
    parser.add_argument("--runMode", default="test")
    parser.add_argument("-m", "--method")
    parser.add_argument("-e", "--epochs")
    parser.add_argument("-d", "--dir")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.cfg is not None:
        cfg = get_config(args.cfg)
    else:
        cfg = default_cfg
    cfg = update_config(cfg, args)
    if args.runMode == "train":
        train(cfg, args.debug)
    else:
        test(cfg)
