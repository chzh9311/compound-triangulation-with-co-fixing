import os

import numpy as np
import cv2
import torch
import pickle
from torch.utils.data import Dataset
from lib.dataset.Camera import Camera
from lib.dataset.human36m import crop_image, generate_gaussian_target, gaussian_direction_field, normalize_box
from lib.utils.DictTree import create_human_tree
from lib.utils.functions import project

from collections import defaultdict, OrderedDict
from easydict import EasyDict as edict

HT = create_human_tree("totalcapture")
LIMB_PAIRS = HT.limb_pairs

class TotalCaptureBaseDataset(Dataset):

    def __init__(self, root_dir, label_dir,
                 image_shape=(256, 256),
                 undistort=True,
                 heatmap_shape=(64, 64),
                 output_type=[],
                 transform=None,
                 is_train=False,
                 crop=True):

        super(TotalCaptureBaseDataset, self).__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.undistort = undistort
        self.image_shape = image_shape
        self.heatmap_shape = heatmap_shape
        self.transform = transform
        self.output_type = output_type
        self.crop = crop
        self.is_train = is_train
        self.flip_pairs = []

        # with open(os.path.join(self.root_dir, 'labels', 'h36m_%s.pkl'%split), 'rb') as lf:
        self.test_subjects = [1, 2, 3, 4, 5]
        self.train_subjects = [1, 2, 3]
        self.action_names = ['rom', 'walking', 'acting', 'running', 'freestyle']
        train_seqs = {'rom':[1,2,3], 'walking':[1,3], 'acting':[1,2], 'running':[], 'freestyle':[1,2]}
        test_seqs = {'rom':[], 'walking':[2], 'acting':[3], 'running':[], 'freestyle':[3]}

    
    def _use_particular_cameras(self, use_cameras, drop_outsiders):
        """
        drop outsiders: 0 means no dropping; 1 means dropping frames fully out of view; 2 means dropping frames part out of view.
        """
        full_labels = self.labels.copy()
        self.labels = []
        for sample in full_labels:
            jp2d = sample['joints_2d']
            jbox = [np.min(jp2d[:, 0]), np.min(jp2d[:, 1]), np.max(jp2d[:, 0]), np.max(jp2d[:, 1])]
            if sample['camera_id'] + 1 in use_cameras:
                # if drop_outsiders == 1 and (sample['box'][2] > 1920 or sample['box'][3] > 1080 or sample['box'][0] < 0 or sample['box'][1] < 0):
                if drop_outsiders == 1 and (jbox[0] > 1920 or jbox[1] > 1080 or jbox[2] < 0 or jbox[3] < 0):
                    pass
                elif drop_outsiders == 2 and (jbox[0] < 0 or jbox[1] < 0 or jbox[2] > 1920 or jbox[3] > 1080):
                    pass
                else:
                    self.labels.append(sample)

    def _sample_frames(self, rate):
        if rate > 1:
            self.labels = [self.labels[i] for i in range(0, len(self.labels), rate)]

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    
class TotalCaptureMonocularFeatureMapDataset(TotalCaptureBaseDataset):
    
    def __init__(self, root_dir, label_dir,
                 sigma=2,
                 image_shape=(256, 256),
                 undistort=True,
                 heatmap_shape=(64, 64),
                 output_type=None,
                 transform=None,
                 is_train=False,
                 frame_sample_rate=1,
                 feature_dim=3,
                 use_cameras=[1, 3, 5, 7],
                 refine_indicator=0,
                 limb_sigmas=[],
                 crop=True):
        super(TotalCaptureMonocularFeatureMapDataset, self).__init__(
            root_dir, label_dir, image_shape, undistort, heatmap_shape, output_type,
            transform, is_train, crop
        )
        self.sigma = sigma
        self.limb_sigmas = []
        self.feature_dim = feature_dim

        split = 'train' if is_train else 'validation'
        with open(os.path.join(self.label_dir, f'totalcapture_{split}.pkl'), 'rb') as lf:
            self.labels= pickle.load(lf)
        
        self._use_particular_cameras(use_cameras, refine_indicator)
        self._sample_frames(frame_sample_rate)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.labels[idx]
        subject = sample['subject']
        action = self.action_names[sample["action"] - 1]
        img_path = os.path.join(self.root_dir, sample['image'])
        assert os.path.isfile(img_path), f'{img_path} doesn\'t exist'

        bbox = sample['box']
        bbox = [int(bbox[i]) for i in [2, 3, 0, 1]] #LTRB
        bbox = normalize_box(bbox)
        image = cv2.imread(img_path)

        K = np.array([[sample['camera']['fx'], 0, sample['camera']['cx']],
                      [0, sample['camera']['fy'], sample['camera']['cy']],
                      [0, 0, 1]])
        camera = Camera(sample['camera']['R'], -1*sample['camera']['R'] @ sample['camera']['T'], K, sample['camera']['name'])

        if self.crop:
            # crop image
            image = crop_image(image, bbox)
            camera.update_after_crop(bbox)
        
        if self.image_shape is not None:
            # resize
            image_shape_before_resize = image.shape[:2]
            image = cv2.resize(image, self.image_shape)
            camera.update_after_resize(image_shape_before_resize, self.image_shape)

        if self.transform:
            image = self.transform(image)

        # heatmaps
        joints_2d = sample['joints_2d'] # 17 x 2
        joint_vis1 = np.logical_and(joints_2d[:, 0] > 0, joints_2d[:, 1] > 0)
        joint_vis2 = np.logical_and(joints_2d[:, 0] < 1920, joints_2d[:, 1] < 1080)
        joint_vis = np.logical_and(joint_vis1, joint_vis2) # 17
        limb_vis = np.logical_and(joint_vis[LIMB_PAIRS[:, 0]], joint_vis[LIMB_PAIRS[:, 1]])
        joints_3d = sample['joints_3d'] # 17 x 3, local coordinates
        for i, j in self.flip_pairs:
            joints_2d[[i, j], :] = joints_2d[[j, i], :]
            joints_3d[[i, j], :] = joints_3d[[j, i], :]

        feat_strides = (bbox[2:] - bbox[:2]) / np.array(self.heatmap_shape)
        kps_in_hm = (joints_2d - bbox[:2].reshape(1, 2)) / feat_strides.reshape(1, 2)
        bones = kps_in_hm[LIMB_PAIRS, :]
        kps_hm = generate_gaussian_target(self.heatmap_shape, kps_in_hm, self.sigma)

        output = []
        for out in self.output_type:
            if out == 'images':
                output.append(image)
            elif out == 'keypoints2d':
                output.append(joints_2d)
            elif out == 'keypoints3d':
                output.append(joints_3d)
            elif out == "keypoints2d_inhm":
                output.append(kps_in_hm)
            elif out == "heatmap":
                output.append(kps_hm)
            elif out == "joint_vis":
                output.append(joint_vis)
            elif out == "limb_vis":
                output.append(limb_vis)
            elif out == "index":
                output.append(idx)
            elif out == "lof":
                dires = joints_3d.T[:, LIMB_PAIRS[:, 1]] - joints_3d.T[:, LIMB_PAIRS[:, 0]]
                dires /= np.linalg.norm(dires, axis=0, keepdims=True)
                if len(self.limb_sigmas):
                    df_hm = gaussian_direction_field(self.heatmap_shape, dires.T, bones, self.limb_sigmas)
                else:
                    df_hm = gaussian_direction_field(self.heatmap_shape, dires.T, bones, [self.sigma] * LIMB_PAIRS.shape[0])
                output.append(df_hm[:, :self.feature_dim].reshape(-1, *self.heatmap_shape))
        
        return output


class TotalCaptureMultiViewDataset(TotalCaptureBaseDataset):
    def __init__(self, root_dir, label_dir,
                 image_shape=(256, 256),
                 undistort=True,
                 heatmap_shape=(64, 64),
                 output_type=[],
                 transform=None,
                 is_train=False,
                 crop=True,
                 use_gt_data_type=None,
                 use_cameras=[1, 3, 5, 7],
                 frame_sample_rate=1,
                 refine_indicator=0,
                 sigma=2):
        super(TotalCaptureMultiViewDataset, self).__init__(
            root_dir, label_dir, image_shape, undistort, heatmap_shape, output_type,
            transform, is_train, crop
        )

        self.n_views = len(use_cameras)
        self.use_gt_data_type = use_gt_data_type
        self.use_cameras = use_cameras
        self.refine_id = refine_indicator
        split = 'train' if is_train else 'validation'
        self.label_file = os.path.join('offlines', 'totalcapture', "multiview",
                                       f"{split}_cam{'-'.join([str(c) for c in self.use_cameras])}_ref{self.refine_id}.pkl")
        if os.path.exists(self.label_file):
            with open(self.label_file, 'rb') as pkf:
                self.labels = pickle.load(pkf)
        else:
            with open(os.path.join(self.label_dir, f'totalcapture_{split}.pkl'), 'rb') as lf:
                self.labels= pickle.load(lf)
            self._use_particular_cameras(use_cameras, refine_indicator)
            self._generate_multi_view_labels(drop_incomplete=True)
        self._sample_frames(frame_sample_rate)
        # self._refine_frames(40)
    
    def _generate_multi_view_labels(self, drop_incomplete=True):
        mono_labels = self.labels.copy()
        group_labels = OrderedDict()
        for sample in mono_labels:
            k = (sample['subject'], sample['action'], sample['subaction'], sample['image_id'])
            if k in group_labels:
                group_labels[k].append(sample)
            else:
                group_labels[k] = [sample]

        self.labels = []

        for k, v in group_labels.items():
            if drop_incomplete and len(v) == self.n_views:
                self.labels.append(v)
        path = os.path.join('offlines', 'totalcapture', "multiview")
        if not os.path.exists(path):
            os.makedirs(path)
        with open(self.label_file, "wb") as pkf:
            pickle.dump(self.labels, pkf)
        

    def _refine_frames(self, th):
        refined = []
        prev_pose3d = np.zeros((HT.size, 3))
        for f in self.labels:
            pose3d = f[0]["joints_gt"]
            if np.max(np.linalg.norm(pose3d - prev_pose3d, axis=1)) > th:
                refined.append(f)
                prev_pose3d = pose3d
        
        self.labels = refined
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.labels[idx]
        frame_data = defaultdict(list)

        for view_data in sample:
            K = np.array([[view_data['camera']['fx'], 0, view_data['camera']['cx']],
                          [0, view_data['camera']['fy'], view_data['camera']['cy']],
                          [0, 0, 1]])
            camera = Camera(view_data['camera']['R'], -1*view_data['camera']['R'] @ view_data['camera']['T'],
                            K, view_data['camera']['name'])

            img_path = os.path.join(self.root_dir, view_data['image'])
            assert os.path.isfile(img_path), f'{img_path} doesn\'t exist'

            bbox = view_data['box']
            bbox = [int(bbox[i]) for i in [2, 3, 0, 1]] #LTRB
            bbox = normalize_box(bbox)
            image = cv2.imread(img_path)

            # limb vis:
            j2d = view_data["joints_2d"]
            joint_vis = (j2d[:, 0] > 0) * (j2d[:, 1] > 0) * (j2d[:, 0] < 1920) * (j2d[:, 1] < 1080)
            limb_vis = joint_vis[LIMB_PAIRS[:, 0]] * joint_vis[LIMB_PAIRS[:, 1]]

            if self.crop:
                # crop image
                image = crop_image(image, bbox)
                camera.update_after_crop(bbox)
            
            if self.image_shape is not None:
                # resize
                image_shape_before_resize = image.shape[:2]
                image = cv2.resize(image, self.image_shape)
                camera.update_after_resize(image_shape_before_resize, self.image_shape)

            if self.transform:
                image = self.transform(image)
            
            frame_data['images'].append(image)
            frame_data['cameras'].append(camera)
            frame_data['proj_matrices'].append(camera.projection)
            frame_data['cam_ctr'].append(-camera.R.T @ camera.t)
            frame_data['intrinsics'].append(camera.K)
            frame_data['rotation'].append(camera.R)
            frame_data['limb_vis'].append(limb_vis)

        keypoints = sample[0]['joints_gt']
        subject = f"S{sample[0]['subject']}"
        for i, j in self.flip_pairs:
            keypoints[[i, j], :] = keypoints[[j, i], :]
        
        images = np.stack(frame_data['images'], axis=0)
        proj_matrices = np.stack(frame_data["proj_matrices"], axis=0)
        intrinsics = np.stack(frame_data['intrinsics'], axis=0)
        rotation = np.stack(frame_data['rotation'], axis=0)
        limb_vis = np.stack(frame_data['limb_vis'], axis=0)
        cam_ctr = np.stack(frame_data['cam_ctr'], axis=0)
        data_id = f"{'Seen' if subject in ['S1', 'S2', 'S3'] else 'Unseen'}, {self.action_names[sample[0]['action']-1]}{sample[0]['subaction']}"

        label2value = {"images": "images", "keypoints3d": "keypoints", "projections":"proj_matrices",
                       "intrinsics": "intrinsics", "rotation": "rotation", "lof": "lof",
                       "identity": "data_id", "limb_vis": "limb_vis", "cam_ctr":"cam_ctr", "index": "idx",
                       "subject": "subject"}
        output = []
        for l in self.output_type:
            output.append(eval(label2value[l]))

        return output