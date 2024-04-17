"""
The dataloader interface to join two datasets.
"""
import os
import sys
import yaml

import numpy as np
import scipy.io as scio
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import Dataset
from lib.dataset.human36m import Human36MMonocularFeatureMapDataset, Human36MMultiViewDataset
from lib.dataset.totalcapture import TotalCaptureMonocularFeatureMapDataset, TotalCaptureMultiViewDataset
from lib.dataset.mhad import MHADHeatmapDataset, MHADStereoDataset

class JointDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        super(JointDataset, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx] + [np.array([0,], dtype=np.uint8)]
        else:
            return self.dataset2[idx - len(self.dataset1)] + [np.array([1,], dtype=np.uint8),]


def build_2D_dataset(cfg, transform, is_train, crop):
    dataset_name = cfg.DATASET.NAME
    if dataset_name == "human3.6m":
        ret_set = Human36MMonocularFeatureMapDataset(
            root_dir=cfg.DATASET.ROOT,
            label_dir=cfg.DATASET.MONOLABELS,
            sigma=cfg.MODEL.EXTRA.SIGMA,
            image_shape=tuple(cfg.MODEL.IMAGE_SIZE),
            heatmap_shape=tuple(cfg.MODEL.EXTRA.HEATMAP_SIZE),
            output_type=cfg.MODEL.REQUIRED_DATA,
            is_train=is_train,
            transform=transform,
            crop=crop
        )
    elif dataset_name == "totalcapture":
        ret_set = TotalCaptureMonocularFeatureMapDataset(
            root_dir=cfg.DATASET.ROOT,
            label_dir=cfg.DATASET.LABELS,
            sigma=cfg.MODEL.EXTRA.SIGMA,
            image_shape=tuple(cfg.MODEL.IMAGE_SIZE),
            heatmap_shape=tuple(cfg.MODEL.EXTRA.HEATMAP_SIZE),
            feature_dim=cfg.MODEL.NUM_DIMS,
            output_type=cfg.MODEL.REQUIRED_DATA,
            is_train=is_train,
            use_cameras=cfg.TRAIN.USE_CAMERAS,
            transform=transform,
            refine_indicator=cfg.TRAIN.REFINE_INDICATOR,
            crop=crop
        )
    elif dataset_name == "mhad":
        ret_set = MHADHeatmapDataset(
            root_dir=cfg.DATASET.ROOT,
            label_dir=cfg.DATASET.LABELS,
            image_shape=tuple(cfg.MODEL.IMAGE_SIZE),
            heatmap_shape=tuple(cfg.MODEL.EXTRA.HEATMAP_SIZE),
            output_type=cfg.MODEL.REQUIRED_DATA,
            transform=transform,
            test_sample_rate=4,
            is_train=is_train,
            rectificated=True,
            baseline=cfg.DATASET.BASELINE,
            crop=crop)
    elif dataset_name == "joint":
        tmp_cfg = deepcopy(cfg)
        tmp_cfg.DATASET = cfg.DATASET1
        set1 = build_2D_dataset(tmp_cfg, transform, is_train, crop)
        tmp_cfg.DATASET = cfg.DATASET2
        set2 = build_2D_dataset(tmp_cfg, transform, is_train, crop)
        ret_set = JointDataset(set1, set2)
    else:
        print(f'No dataset named {dataset_name}')
        exit(-1)

    return ret_set


def build_3D_dataset(cfg, transform, is_train, crop):
    dataset_name = cfg.DATASET.NAME
    if dataset_name == "human3.6m":
        ret_set = Human36MMultiViewDataset(
            root_dir=cfg.DATASET.ROOT,
            label_dir=cfg.DATASET.LABELS,
            image_shape=tuple(cfg.MODEL.IMAGE_SIZE),
            is_train=is_train,
            transform=transform,
            crop=crop,
            output_type=cfg.MODEL.REQUIRED_DATA,
            use_cameras=cfg.TRAIN.USE_CAMERAS if is_train else cfg.TEST.USE_CAMERAS,
            stereo_sample=cfg.TRAIN.STEREO_SAMPLE if is_train else cfg.TEST.STEREO_SAMPLE,
            with_damaged_actions=cfg.DATASET.WITH_DAMAGED_ACTIONS,
            sigma=cfg.MODEL.EXTRA.SIGMA,
        )
    elif dataset_name == "totalcapture":
        train_set = TotalCaptureMultiViewDataset(
            root_dir=cfg.DATASET.ROOT,
            label_dir=cfg.DATASET.LABELS,
            image_shape=tuple(cfg.MODEL.IMAGE_SIZE),
            is_train=True,
            transform=transform,
            crop=True,
            output_type=cfg.MODEL.REQUIRED_DATA,
            use_cameras=cfg.TRAIN.USE_CAMERAS,
            frame_sample_rate=1 if is_train else cfg.TEST.FRAME_SAMPLE_RATE,
            refine_indicator=cfg.TRAIN.REFINE_INDICATOR,
            sigma=cfg.MODEL.EXTRA.SIGMA
        )
    elif dataset_name == "mhad":
        ret_set = MHADStereoDataset(
            root_dir=cfg.DATASET.ROOT,
            label_dir=cfg.DATASET.LABELS,
            image_shape=tuple(cfg.MODEL.IMAGE_SIZE),
            output_type=cfg.MODEL.REQUIRED_DATA,
            transform=transform,
            test_sample_rate=4,
            is_train=is_train,
            rectificated=True,
            baseline=cfg.DATASET.BASELINE,
            crop=crop)
    elif dataset_name == "joint":
        tmp_cfg = deepcopy(cfg)
        tmp_cfg.DATASET = cfg.DATASET1
        set1 = build_3D_dataset(tmp_cfg, transform, is_train, crop)
        tmp_cfg.DATASET = cfg.DATASET2
        set2 = build_3D_dataset(tmp_cfg, transform, is_train, crop)
        ret_set = JointDataset(set1, set2)
    else:
        print(f'No dataset named {dataset_name}')
        exit(-1)

    return ret_set
