"""
The dataloader interface for MPI-INF-3DHP dataset
The preprocess is done by https://github.com/CHUNYUWANG/H36M-Toolbox
"""
import os
import sys
import yaml

import numpy as np
import cv2
import scipy.io as scio
from torch.utils.data import Dataset
from lib.dataset.Camera import Camera
from lib.dataset.human36m import *
from lib.utils.DictTree import create_human_tree
from lib.utils.functions import project

from collections import defaultdict
from easydict import EasyDict as edict

ht = create_human_tree("human3.6m")
LIMB_PAIRS = ht.limb_pairs

## Correspondence with H36M
# joint_idx_train_matlab = [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]    # notice: it is in matlab index
# joint_idx_train = [i-1 for i in joint_idx_train_matlab]
joint_idx_train = [25, 24, 23, 18, 19, 20, 4, 2, 5, 7, 16, 15, 14, 9, 10, 11, 6]

class MPI_INF_3DHP_train(Dataset):
    """
    The base class for MPI_INF_3DPH dataset interface.
    """
    def __init__(self, root_dir, calib_dir,
                 image_shape=(256, 256),
                 heatmap_shape=(64, 64),
                 output_type=[],
                 transform=None,
                 sigma=2,
                 refine_labels=True,
                 crop=True):
        self.root_dir = root_dir
        self.calib_dir = calib_dir
        self.image_shape = image_shape
        self.heatmap_shape = heatmap_shape
        self.transform = transform
        self.output_type = output_type
        self.sigma = sigma
        self.crop = crop
        self.flip_pairs = []

        self.n_joints = 17

        train_subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
        cameras = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # labels = np.load(os.path.join(self.label_dir, "data_3d_mpi_inf_3dhp_data.npz"), allow_pickle=True)
        self.labels = []
        for s in train_subjects:
            for seq in ["Seq1", "Seq2"]:
                hf = scio.loadmat(os.path.join(self.root_dir, s, seq, "annot.mat"))
                for c in cameras:
                    positions_3d = hf["annot3"][c, 0].reshape(-1, 28, 3)
                    positions_2d = hf["annot2"][c, 0].reshape(-1, 28, 2)
                    positions_3d = positions_3d[:, joint_idx_train, :].astype(np.float32)
                    positions_2d = positions_2d[:, joint_idx_train, :].astype(np.float32)
                    # positions_3d = positions_3d.astype(np.float32)
                    # positions_2d = positions_2d.astype(np.float32)

                    # generate bbox
                    lt = np.min(positions_2d, axis=1) - 60
                    rb = np.max(positions_2d, axis=1) + 60
                    bbox = np.concatenate((lt, rb), axis=1)

                    for idx in hf["frames"]:
                        idx = idx.item()
                        # Delete frames that does not exists
                        if idx + 1 == 12489 and s == "S3" and seq == "Seq1":
                            continue
                        # In case some joints out of sight:
                        if bbox[idx, 0] < 0 or bbox[idx, 1] < 0 or bbox[idx, 2] > 2048 or bbox[idx, 3] > 2048:
                            continue
                        frame = {}
                        frame["subject"] = s
                        frame["sequence"] = seq
                        frame["cam_id"] = c
                        frame["frame_id"] = idx
                        frame["bbox"] = bbox[idx]
                        frame["positions_3d"] = positions_3d[idx, ...]
                        frame["positions_2d"] = positions_2d[idx, ...]
                        self.labels.append(frame)
        
        if refine_labels:
            self.refine_frames(40)
        
        with open(os.path.join(self.calib_dir, "cam_params.yaml"), "r") as f:
            self.calib = yaml.load(f, Loader=yaml.FullLoader)
        
        for c in cameras:
            self.calib[f"cam_{c}"]["R"] = self.quaternion2rot(self.calib[f"cam_{c}"]["extrinsics"]["orientation"])
            focals = self.calib[f"cam_{c}"]["intrinsics"]["focal_length"]
            center = self.calib[f"cam_{c}"]["intrinsics"]["center"]
            self.calib[f"cam_{c}"]["K"] = np.array([[focals[0], 0, center[0]], [0, focals[1], center[1]], [0, 0, 1]])
            self.calib[f"cam_{c}"]["t"] = self.calib[f"cam_{c}"]["extrinsics"]["translation"]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        frame = self.labels[idx]
        img_path = os.path.join(self.root_dir, frame["subject"], frame["sequence"], "imageFrames",
                               f"video_{frame['cam_id']}", f"frame_{frame['frame_id']+1:06d}.jpg")
        image = cv2.imread(img_path)
        joints_3d = frame["positions_3d"]
        joints_2d = frame["positions_2d"]
        bbox = normalize_box(frame["bbox"])
        cam = self.calib[f'cam_{frame["cam_id"]}']
        camera = Camera(cam['R'], cam['t'], cam['K'], None, frame["cam_id"])

        if self.crop:
            image = crop_image(image, bbox)
            camera.update_after_crop(bbox)
        
        if self.image_shape is not None:
            image_shape_before_resize = image.shape[:2]
            image = cv2.resize(image, self.image_shape)
            camera.update_after_resize(image_shape_before_resize, self.image_shape)

        if self.transform:
            image = self.transform(image)

        feat_strides = (bbox[2:] - bbox[:2]) / np.array(self.heatmap_shape)
        kps_in_hm = (joints_2d - bbox[:2].reshape(1, 2)) / feat_strides.reshape(1, 2)

        # In MPI-INF-3DHP, 3D coordinates are recorded locally.
        # local_coords = (camera.R @ joints_3d.T).T + camera.t.T
        # kps_in_hm = (camera.K @ local_coords.T).T
        # kps_in_hm = kps_in_hm[:, :2] / kps_in_hm[:, 2:3] / 4
        # kps_in_hm = joints_3d @ camera.K.T
        # kps_in_hm = kps_in_hm[:, :2] / kps_in_hm[:, 2:3] / 4

        kps_hm = generate_gaussian_target(self.heatmap_shape, kps_in_hm, self.sigma)

        depths = (np.concatenate((joints_3d, np.ones((self.n_joints, 1))), axis=1) @ camera.projection[2:3, :].T).squeeze()
        mus = depths[LIMB_PAIRS[:, 1]] / depths[LIMB_PAIRS[:, 0]]
        bones = kps_in_hm[LIMB_PAIRS, :]
        output = []

        for out in self.output_type:
            if out == "images":
                output.append(image)
            elif out == "keypoints2d":
                output.append(joints_2d)
            elif out == "keypoints2d_inhm":
                output.append(kps_in_hm)
            elif out == "heatmap":
                output.append(kps_hm)
            elif out == "keypoints3d":
                output.append(joints_3d)
            elif out == "mu":
                output.append(mus)
            elif out == "directionmap":
                local_coords = joints_3d.T
                dires = local_coords[:, LIMB_PAIRS[:, 1]] - local_coords[:, LIMB_PAIRS[:, 0]]
                dires /= np.linalg.norm(dires, axis=0, keepdims=True)
                df_hm = gaussian_direction_field(self.heatmap_shape, dires.T, bones, self.sigma)
                output.append(df_hm.reshape(-1, *self.heatmap_shape))
            elif out == "densitymap2d":
                df_hm = gaussian_density_field(self.heatmap_shape, mus, bones, self.sigma ** 2)
                output.append(df_hm)
            # elif out == "densitymap_new_weak":
            #     new_df = generate_new_density_field(self.heatmap_shape, dires.T, bones, cam["K"], True)
            #     output.append(new_df)
            # elif out == "densitymap_new":
            #     new_df = generate_new_density_field(self.heatmap_shape, dires.T, bones, cam["K"], False)
            #     output.append(new_df)

        return output

    def refine_frames(self, th):
        refined = []
        prev_pose3d = np.zeros((self.n_joints, 3))
        for f in self.labels:
            pose3d = f["positions_3d"]
            if np.max(np.linalg.norm(pose3d - prev_pose3d, axis=1)) > th:
                refined.append(f)
                prev_pose3d = pose3d
        
        self.labels = refined
    
    def quaternion2rot(self, q):
        x, y, z, w = q
        rot = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y + 2*w*z, 2*x*z - 2*w*y],
                        [2*x*y - 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z + 2*w*x],
                        [2*x*z + 2*w*y, 2*y*z - 2*w*x, 1 - 2*x**2 - 2*y**2]], dtype=np.float32)
            
        return rot
        

class H36M3DHPJointDataset(Dataset):
    """
    Used for joint training.
    """
    def __init__(self, h36m_root_dir, h36m_label_dir,
                 mpi3d_root_dir, mpi3d_calib_dir,
                 sigma=2,
                 image_shape=(256, 256),
                 undistort=True,
                 heatmap_shape=(64, 64),
                 output_type=[],
                 sample_level=2,
                 refine_labels=True,
                 transform=None,
                 with_damaged_actions=True,
                 is_train=False,
                 crop=True):
        self.h36m = Human36MMonocularFeatureMapDataset(
            h36m_root_dir, h36m_label_dir, sigma, image_shape, undistort, heatmap_shape,
            output_type, sample_level, transform, with_damaged_actions, is_train, crop
        )
        if is_train:
            self.mpi3d = MPI_INF_3DHP_train(
                root_dir=mpi3d_root_dir, calib_dir=mpi3d_calib_dir, image_shape=image_shape,
                heatmap_shape=heatmap_shape, output_type=output_type, refine_labels=refine_labels,
                transform=transform, crop=crop
            )
        self.order = np.arange(len(self.h36m.labels["table"]) + len(self.mpi3d.labels))
        
    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        idx = self.order[idx]
        if idx < len(self.h36m):
            return self.h36m[idx] + [0,]
        else:
            return self.mpi3d[idx - len(self.h36m)] + [1,]

    def set_sample_order(self, order):
        self.order = order