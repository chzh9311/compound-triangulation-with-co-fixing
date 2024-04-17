'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-03-14 21:32:03
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-09-09 13:52:34
FilePath: /wxy/3d_pose/stereo-estimation/lib/dataset/mhad_stereo.py
Description: stereo view dataset of  MHAD_Berkeley
'''
import os
from collections import defaultdict
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from lib.dataset.Camera import Camera
from lib.utils.DictTree import create_human_tree
# from lib.utils.img import resize_image, normalize_image, scale_bbox, crop_keypoints_img, resize_keypoints_img
from lib.dataset.human36m import crop_image, generate_gaussian_target, gaussian_lof

ht = create_human_tree("mhad")
LIMB_PAIRS = ht.limb_pairs

class MHADBaseDataset(Dataset):
    def __init__(self, root_dir, label_dir,
                 image_shape=(256, 256),
                 heatmap_shape=(64, 64),
                 output_type=[],
                 transform=None,
                 test_sample_rate=1,
                 is_train=False,
                 rectificated=True,
                 baseline='s',
                 crop=True):

        self.root_dir = root_dir
        self.label_dir = label_dir
        self.image_shape = image_shape
        self.transform = transform
        self.output_type = output_type
        self.crop = crop
        self.is_train = is_train
        self.rectificated = rectificated
        self.baseline_width=baseline
        # self.flip_pairs = [[3, 5], [10, 12]]
        self.flip_pairs = []
        self.num_joints = 17

        self.labels = np.load(os.path.join(self.label_dir, f'mhad-stereo-{baseline}-labels-GTbboxes.npy'), allow_pickle=True).item()

        train_subjects = ['S01','S02','S03','S04','S05','S06','S07','S09','S10','S12']
        test_subjects = ['S08', 'S11']
        train_subjects = list(self.labels['subject_names'].index(x)
                                for x in train_subjects)
        test_subjects = list(self.labels['subject_names'].index(x)
                                for x in test_subjects)

        if is_train:
            mask = np.isin(self.labels['table']['subject_idx'],
                           train_subjects,
                           assume_unique=True)
            indices = np.nonzero(mask)[0]
        else:
            mask = np.isin(self.labels['table']['subject_idx'],
                           test_subjects,
                           assume_unique=True)
            indices = np.nonzero(mask)[0][::test_sample_rate]

        self.labels['table'] = self.labels['table'][indices]
    
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    
class MHADHeatmapDataset(MHADBaseDataset):
    """
        MHAD_Berkeley for stereoview tasks
    """
    def __init__(self, root_dir, label_dir,
                 image_shape=(256, 256),
                 heatmap_shape=(64, 64),
                 output_type=[],
                 transform=None,
                 test_sample_rate=1,
                 is_train=False,
                 rectificated=True,
                 sigma=2,
                 baseline='s',
                 crop=True):

        super(MHADHeatmapDataset, self).__init__(
            root_dir, label_dir,
            image_shape=image_shape,
            output_type=output_type,
            transform=transform,
            test_sample_rate=test_sample_rate,
            is_train=is_train,
            rectificated=rectificated,
            baseline=baseline,
            crop=crop)
        self.sigma = sigma
        self.heatmap_shape=heatmap_shape,


    def __len__(self):
        # 2 views
        return len(self.labels['table']) * 2


    def __getitem__(self, idx):
        shot = self.labels['table'][int(idx//2)]

        group_idx = shot['group_idx']
        subject_idx = shot['subject_idx']
        action_idx = shot['action_idx']
        reptition_idx = shot['reptition_idx']
        subject = self.labels['subject_names'][subject_idx]
        action = self.labels['action_names'][action_idx]
        reptition = self.labels['reptition_names'][reptition_idx]
        frame_idx = shot['frame_idx']

        camera_idx = idx % 2
        camera_name = self.labels['camera_names'][group_idx][camera_idx]
        
        # if self.action_target is not None and self.action_target not in action:
            # return None
        
        # keypoints_3d = np.pad(shot['keypoints'][:self.num_joints], ((0, 0), (0, 1)),
        #                       'constant', constant_values=1.0)
        # load bounding box
        bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1, 0, 3, 2]]  # TLBR to LTRB
        bbox_height = bbox[2] - bbox[0]
        if bbox_height == 0:
            # convention: if the bbox is empty, then this view is missing
            return

        # scale the bounding box = 1.5
        # bbox = scale_bbox(bbox, self.scale_bbox)

        # load image
        image_path = os.path.join(
            self.root_dir, 'Cluster01', 'rectificated' * self.rectificated, 
            self.baseline_width * self.rectificated, str(group_idx) * self.rectificated,
            'Cam%02d'%(int(camera_name)), subject, action, reptition, 
            'img_l01_c%02d_s%02d_a%02d_r%02d_%05d.jpg' % (int(camera_name), subject_idx+1, action_idx+1, reptition_idx+1, frame_idx))
        
        assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
        image = cv2.imread(image_path)
        bbox0 = bbox.copy()

        # load camera
        shot_camera = self.labels['cameras'][group_idx][camera_idx]
        retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'],
                                name = camera_name)
        
        if self.crop:
            # crop image
            image = crop_image(image, bbox)
            retval_camera.update_after_crop(bbox)

        if self.image_shape is not None:
            # resize
            image_shape_before_resize = image.shape[:2]
            image = cv2.resize(image, self.image_shape)
            retval_camera.update_after_resize(image_shape_before_resize, self.image_shape)
        image = self.transform(image)
            
        joints_2d = shot['keypoints_2d'][camera_idx]
        joints_3d = shot['keypoints']
        for i, j in self.flip_pairs:
            joints_2d[[i, j], :] = joints_2d[[j, i], :]
            joints_3d[[i, j], :] = joints_3d[[j, i], :]

        joints_vis = joints_2d[..., -1]
        joints_2d = joints_2d[..., :2]
        feat_strides = (bbox0[2:] - bbox0[:2]) / np.array(self.heatmap_shape)
        kps_in_hm = (joints_2d - bbox0[:2].reshape(1, 2)) / feat_strides.reshape(1, 2)
        kps_hm = generate_gaussian_target(self.heatmap_shape, kps_in_hm, self.sigma)

        depths = (np.concatenate((joints_3d, np.ones((17, 1))), axis=1) @ retval_camera.projection[2:3, :].T).squeeze()
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
            elif out == "mu":
                output.append(mus)
            elif out == "lof":
                local_coords = retval_camera.R @ joints_3d.T + retval_camera.t
                dires = local_coords[:, LIMB_PAIRS[:, 1]] - local_coords[:, LIMB_PAIRS[:, 0]]
                dires /= np.linalg.norm(dires, axis=0, keepdims=True)
                df_hm = gaussian_lof(self.heatmap_shape, dires.T, bones, [self.sigma]*LIMB_PAIRS.shape[0])
                output.append(df_hm.reshape(-1, *self.heatmap_shape))
            elif out == "joint_vis":
                output.append(np.ones(joints_2d.shape[0]))
            elif out == "limb_vis":
                output.append(np.ones(joints_2d.shape[0]-1))

        return output


class MHADStereoDataset(MHADBaseDataset):
    def __init__(self, root_dir, label_dir,
                 image_shape=(256, 256),
                 output_type=[],
                 transform=None,
                 test_sample_rate=1,
                 is_train=False,
                 rectificated=True,
                 baseline='s',
                 crop=True):

        super(MHADStereoDataset, self).__init__(
            root_dir, label_dir,
            image_shape=image_shape,
            output_type=output_type,
            transform=transform,
            test_sample_rate=test_sample_rate,
            is_train=is_train,
            rectificated=rectificated,
            baseline=baseline,
            crop=crop)

    def __len__(self):
        return len(self.labels['table'])
    
    def __getitem__(self, idx):
        sample = defaultdict(list)
        shot = self.labels['table'][idx]

        group_idx = shot['group_idx']
        subject_idx = shot['subject_idx']
        action_idx = shot['action_idx']
        reptition_idx = shot['reptition_idx']
        subject = self.labels['subject_names'][subject_idx]
        action = self.labels['action_names'][action_idx]
        reptition = self.labels['reptition_names'][reptition_idx]
        frame_idx = shot['frame_idx']
        
        # if self.action_target is not None and self.action_target not in action:
        #     return None
        
        for camera_idx, camera_name in enumerate(self.labels['camera_names'][group_idx]):
            # load bounding box
            bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1, 0, 3, 2]]  # TLBR to LTRB
            bbox_height = bbox[2] - bbox[0]

            # scale the bounding box
            # bbox = scale_bbox(bbox, self.scale_bbox)

            # load image
            image_path = os.path.join(
                self.root_dir, 'Cluster01', 'rectificated' * self.rectificated, 
                self.baseline_width * self.rectificated, str(group_idx) * self.rectificated,
                'Cam%02d'%(int(camera_name)), subject, action, reptition, 
                'img_l01_c%02d_s%02d_a%02d_r%02d_%05d.jpg' % (int(camera_name), subject_idx+1, action_idx+1, reptition_idx+1, frame_idx))
            
            assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
            image = cv2.imread(image_path)

            # load camera
            shot_camera = self.labels['cameras'][group_idx][camera_idx]
            retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], name = camera_name)

            if self.crop and bbox_height > 0:
                # crop image
                image = crop_image(image, bbox)
                retval_camera.update_after_crop(bbox)

            if self.image_shape is not None:
                # resize
                image_shape_before_resize = image.shape[:2]
                image = cv2.resize(image, self.image_shape)
                retval_camera.update_after_resize(image_shape_before_resize, self.image_shape)

                sample['image_shapes_before_resize'].append(image_shape_before_resize)
            
            if self.transform:
                image = self.transform(image)

            sample['images'].append(image)
            sample['detections'].append(bbox + (1.0,))
            sample['cameras'].append(retval_camera)
            sample['proj_matrices'].append(retval_camera.projection)
            sample['intrinsics'].append(retval_camera.K)
            sample['cam_ctr'].append(- retval_camera.R.T @ retval_camera.t)
            sample['rotation'].append(retval_camera.R)

        # 3D keypoints
        sample['keypoints'] = shot['keypoints']
        for i, j in self.flip_pairs:
            sample['keypoints'][[i, j], :] = sample['keypoints'][[j, i], :]

        # save sample's index
        sample['indexes'] = idx

        # Post-process to output
        images = np.stack(sample["images"], axis=0)
        keypoints = sample["keypoints"]
        proj_matrices = np.stack(sample["proj_matrices"], axis=0)
        intrinsics = np.stack(sample['intrinsics'], axis=0)
        rotation = np.stack(sample['rotation'], axis=0)
        cam_ctr = np.stack(sample['cam_ctr'], axis=0)
        # mus, density_map, bvs = self.generate_gt_density(proj_matrices, keypoints, 2)
        label2value = {"images": "images", "mus": "mus", "keypoints3d": "keypoints", "projections":"proj_matrices",
                       "densitymap2d": "density_map", "bonevectors": "bvs", "intrinsics": "intrinsics",
                       "rotation": "rotation", "lof": "lof", "identity": "data_id",
                       "cam_ctr": "cam_ctr", "index": 'idx'}
        output = []
        for l in self.output_type:
            if l == "identity":
                output.append(action)
            elif l == "subject":
                output.append(subject)
            # elif l == "lof":
            #     feat_strides = (bbox0[2:] - bbox0[:2]) / np.array(self.heatmap_shape)
            #     kps_in_hm = (joints_2d - bbox0[:2].reshape(1, 2)) / feat_strides.reshape(1, 2)
            #     kps_hm = generate_gaussian_target(self.heatmap_shape, kps_in_hm, self.sigma)
            elif l == "limb_vis":
                output.append(np.ones((2, LIMB_PAIRS.shape[0])))
            else:
                output.append(eval(label2value[l]))

        return output
