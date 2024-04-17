"""
The dataloader interface for Human3.6M dataset
The preprocess is done by https://github.com/CHUNYUWANG/H36M-Toolbox
"""
import os

from random import shuffle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from lib.dataset.Camera import Camera
from lib.utils.DictTree import create_human_tree
from lib.utils.functions import project

from collections import defaultdict
from easydict import EasyDict as edict

# import warnings
# warnings.filterwarnings('error')

ht = create_human_tree("human3.6m")
LIMB_PAIRS = ht.limb_pairs

class Human36MBaseDataset(Dataset):
    """
    The base class for human3.6M dataset interface.
    """
    def __init__(self, root_dir, label_dir,
                 image_shape=(256, 256),
                 undistort=True,
                 heatmap_shape=(64, 64),
                 sample_level=2,
                 output_type=[],
                 with_damaged_actions=False,
                 transform=None,
                 is_train=False,
                 crop=True):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.undistort = undistort
        self.image_shape = image_shape
        self.heatmap_shape = heatmap_shape
        self.sample_level = sample_level
        self.transform = transform
        self.output_type = output_type
        self.crop = crop
        self.is_train = is_train
        self.flip_pairs = []

        split = 'train' if is_train else 'validation'
        # with open(os.path.join(self.root_dir, 'labels', 'h36m_%s.pkl'%split), 'rb') as lf:
        with open(self.label_dir, 'rb') as lf:
            self.labels= np.load(lf, allow_pickle=True).item()

        train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        test_subjects = ['S9', 'S11']

        train_subjects = [self.labels['subject_names'].index(x) for x in train_subjects]
        test_subjects = [self.labels['subject_names'].index(x) for x in test_subjects]

        indices = []
        if is_train:
            mask = np.isin(self.labels['table']['subject_idx'], train_subjects, assume_unique=True)
            indices.append(np.nonzero(mask)[0])
        else:
            mask = np.isin(self.labels['table']['subject_idx'], test_subjects, assume_unique=True)

            if not with_damaged_actions:
                mask_S9 = self.labels['table']['subject_idx'] == self.labels['subject_names'].index('S9')

                damaged_actions = 'Greeting-2', 'SittingDown-2', 'Waiting-1'
                damaged_actions = [self.labels['action_names'].index(x) for x in damaged_actions]
                mask_damaged_actions = np.isin(self.labels['table']['action_idx'], damaged_actions)

                mask &= ~(mask_S9 & mask_damaged_actions)

            indices.append(np.nonzero(mask)[0])

        self.labels['table'] = self.labels['table'][np.concatenate(indices)]
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError
        

class Human36MMonocularFeatureMapDataset(Human36MBaseDataset):
    """
    The human3.6M dataset interface for pretraining.
    Returns 2D joint feature map and density field map.
    """
    def __init__(self, root_dir, label_dir,
                 sigma=2,
                 image_shape=(256, 256),
                 undistort=True,
                 heatmap_shape=(64, 64),
                 output_type=None,
                 sample_level=2,
                 transform=None,
                 with_damaged_actions=True,
                 is_train=False,
                 limb_sigmas=[],
                 crop=True):
        super(Human36MMonocularFeatureMapDataset, self).__init__(
            root_dir, label_dir, image_shape, undistort, heatmap_shape, sample_level, output_type,
            with_damaged_actions, transform, is_train, crop
        )
        self.sigma = sigma
        self.limb_sigmas=[]

    def __len__(self):
        return len(self.labels['table'])
        # return 200

    def __getitem__(self, idx):
        sample = {}
        shot = self.labels['table'][idx]

        subject = self.labels['subject_names'][shot['subject_idx']]
        action = self.labels['action_names'][shot['action_idx']]
        frame_idx = shot['frame_idx']

        # load bounding box
        camera_name = self.labels["camera_names"][shot["camera_idx"]]
        bbox = shot['bbox_by_camera_tlbr'][[1,0,3,2]]  # TLBR to LTRB
        bbox = normalize_box(bbox)

        # load image
        image_path = os.path.join(
            self.root_dir, 'processed', '_undistorted' * self.undistort, subject,
            action, 'imageSequence' + '-undistorted'*self.undistort,
            camera_name, 'img_%06d.jpg' % (frame_idx+1))
        assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
        image = cv2.imread(image_path)

        # load camera
        shot_camera = self.labels['cameras'][shot['subject_idx'], shot['camera_idx']]
        retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], camera_name)
        bbox0 = bbox.copy()

        if self.crop:
            # crop image
            image = crop_image(image, bbox)
            retval_camera.update_after_crop(bbox)

        if self.image_shape is not None:
            # resize
            image_shape_before_resize = image.shape[:2]
            image = cv2.resize(image, self.image_shape)
            retval_camera.update_after_resize(image_shape_before_resize, self.image_shape)

        if self.transform:
            image = self.transform(image)

        # generate heatmaps
        joints_2d = shot['keypoints_2d']
        joints_3d = shot['keypoints_3d']
        for i, j in self.flip_pairs:
            joints_2d[[i, j], :] = joints_2d[[j, i], :]
            joints_3d[[i, j], :] = joints_3d[[j, i], :]

        feat_strides = (bbox0[2:] - bbox0[:2]) / np.array(self.heatmap_shape)
        kps_in_hm = (joints_2d - bbox0[:2].reshape(1, 2)) / feat_strides.reshape(1, 2)
        kps_hm = generate_gaussian_target(self.heatmap_shape, kps_in_hm, self.sigma)

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
            elif out == "lof":
                local_coords = retval_camera.R @ joints_3d.T + retval_camera.t
                dires = local_coords[:, LIMB_PAIRS[:, 1]] - local_coords[:, LIMB_PAIRS[:, 0]]
                dires /= np.linalg.norm(dires, axis=0, keepdims=True)
                if len(self.limb_sigmas):
                    df_hm = gaussian_lof(self.heatmap_shape, dires.T, bones, self.limb_sigmas)
                else:
                    df_hm = gaussian_lof(self.heatmap_shape, dires.T, bones, [self.sigma]*LIMB_PAIRS.shape[0])
                output.append(df_hm.reshape(-1, *self.heatmap_shape))
            elif out == "densitymap2d":
                df_hm = gaussian_density_field(self.heatmap_shape, mus, bones, self.sigma ** 2)
                output.append(df_hm)
            elif out == "densitymap1d":
                density1d = generate_1d_density(self.density_size, mus)
                output.append(density1d)
            elif out == "joint_vis":
                output.append(np.ones(joints_2d.shape[0]))
            elif out == "limb_vis":
                output.append(np.ones(joints_2d.shape[0]-1))

        return output


def generate_new_density_field(heatmap_shape, dires, bones, K, sigma, weak):
    """
    dires: n_limbs x 3
    bones: n_limbs x 2 x 2
    weak: whether it's weak perspective or not
    """
    device = bones.device
    conf = generate_bone_confidence(heatmap_shape, bones, sigma)
    angles = torch.zeros_like(conf, device=device)
    theta = torch.arctan2(torch.norm(dires[:, :2], dim=1), dires[:, 2])
    if weak:
        angles[conf > 0] = theta.view(-1, 1, 1)
    else:
        prox = torch.concat((bones[:, 0, :], torch.ones(bones.shape[0], 1, device=device)), dim=1)
        dp = - (torch.linalg.inv(K) @ prox.T).T
        bv = torch.concat((bones[:, 1, :] - bones[:, 0, :], torch.zeros(bones.shape[0], 1, device=device)), dim=1)
        beta = torch.arccos(torch.sum(dp * bv, dim=1) / torch.norm(dp, dim=1) / torch.norm(bv, dim=1))

    return torch.stack((conf, angles), dim=1)


def generate_bone_confidence(heatmap_shape, bones, sigma):
    h, w = heatmap_shape
    sigmas = np.array(sigma).reshape(-1, 1, 1)
    grid = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='xy'), axis=0)
    dvs = grid.reshape(1, 2, h, w) - bones[:, 0, :].reshape(-1, 2, 1, 1)
    bvs = bones[:, 1, :] - bones[:, 0, :]
    l = np.linalg.norm(bvs, axis=1)
    bvs = bvs / l.reshape(-1, 1)
    pos = np.sum(dvs * bvs.reshape(-1, 2, 1, 1), axis=1)

    bns = np.stack((-bvs[:, 1], bvs[:, 0]), axis=1)
    dist2 = np.square(np.sum(dvs * bns.reshape(-1, 2, 1, 1), axis=1))

    multiplier = np.exp(-dist2/(2*sigmas**2))
    # 3 sigma principal
    multiplier[dist2 > 9 * sigmas**2] = 0
    dist2p1 = np.square(np.linalg.norm(grid.reshape(1, 2, h, w) - bones[:, 0, :].reshape(-1, 2, 1, 1), axis=1))
    dist2p2 = np.square(np.linalg.norm(grid.reshape(1, 2, h, w) - bones[:, 1, :].reshape(-1, 2, 1, 1), axis=1))
    outerp1 = np.exp(-dist2p1/(2*sigmas**2))
    outerp2 = np.exp(-dist2p2/(2*sigmas**2))
    outerp1[dist2p1 > 9 * sigmas**2] = 0
    outerp2[dist2p2 > 9 * sigmas**2] = 0

    multiplier[pos < 0] = outerp1[pos < 0]
    multiplier[pos - l.reshape(-1, 1, 1) > 0] = outerp2[pos - l.reshape(-1, 1, 1) > 0]

    return multiplier


def gaussian_lof(heatmap_shape, dires, bones, sigma):
    """
    dires: n_limbs x 3
    bones: n_limbs x 2 x 2
    sigma: list
    """
    h, w = heatmap_shape
    multiplier = generate_bone_confidence(heatmap_shape, bones, sigma)

    # return hm / np.sum(hm, axis=(1, 2)).reshape(-1, 1, 1)
    return multiplier.reshape(-1, 1, h, w) * dires.reshape(-1, 3, 1, 1)


def generate_lof(heatmap_shape, dires, bones, limbw):
    """
    dires: n_limbs x 3
    bones: n_limbs x 2 x 2
    dires: n_limbs x 3
    return: n_limbs x 3 x h x w
    """
    height, width = heatmap_shape
    Nb = dires.shape[0]
    grid = np.stack(np.meshgrid(np.arange(height), np.arange(width), indexing='xy'), axis=0).reshape(1, 2, height, width)
    multiplier = np.zeros((Nb, 1, height, width))
    bvs = bones[:, 1, :] - bones[:, 0, :]
    l = np.linalg.norm(bvs, axis=1)
    bvs = bvs / l.reshape(-1, 1)
    vbvs = np.stack((bvs[:, 1], -bvs[:, 0]), axis=1)
    hpos = np.sum(bvs.reshape(-1, 2, 1, 1) * (grid - bones[:, 0, :].reshape(-1, 2, 1, 1)), axis=1, keepdims=True)
    vpos = np.sum(vbvs.reshape(-1, 2, 1, 1) * (grid - bones[:, 0, :].reshape(-1, 2, 1, 1)), axis=1, keepdims=True)
    multiplier[(hpos > -1) * (hpos < l.reshape(-1, 1, 1, 1)+1) * (vpos > -limbw) * (vpos < limbw)] = 1

    return multiplier * dires.reshape(Nb, -1, 1, 1)


def generate_vanishing_map(heatmap_shape, mus, bones, l, density_type, sigma=2):
    """
    generate the vanishing heatmap, where the distance is mapped to [0, 1).
    mus: n_limbs
    bones: n_limbs x 2 x 2
    return: n_limbs x 2 x h x w
    """
    height, width = heatmap_shape
    grid = np.stack(np.meshgrid(np.arange(height), np.arange(width), indexing='xy'), axis=0).reshape(1, 2, height, width)
    bvs = bones[:, 1, :] - bones[:, 0, :]

    # following are all homogeneous.
    # Calculate vanishing point with proximal point.
    w1 = mus.reshape(-1, 1) - 1
    v = mus.reshape(-1, 1) * bvs + w1 * bones[:, 0, :]
    vanish_pts = np.concatenate((v, w1), axis=1)
    offset = v.reshape(-1, 2, 1, 1) - w1.reshape(-1, 1, 1, 1) * grid
    d_inf = np.linalg.norm(offset, axis=1).reshape(-1, 1, height, width)
    offset = offset / d_inf

    w2 = w1.reshape(-1, 1, 1, 1)
    if density_type == "lambda":
        # calculate lambda
        lams = w2 * l / (np.sqrt(d_inf**2 + w2**2 * l**2) + d_inf)
        return lams * offset, vanish_pts
    elif density_type == "mu":
        pmus = d_inf / (np.sqrt(d_inf**2 + w2**2 * l**2) + np.abs(w2) * l)
        return pmus * offset, vanish_pts
    elif density_type == "rt":
        w = vanish_pts[:, 2]
        vanish_pts /= np.linalg.norm(vanish_pts, axis=1, keepdims=True)
        rho = np.sqrt(1 - w ** 2)
        r0 = rho / (np.sqrt(rho**2 + w ** 2 * l ** 2) + np.abs(w) * l) # 0 ~ 1
        theta = np.arctan2(vanish_pts[:, 1], vanish_pts[:, 0]) # -pi ~ pi
        r0[w < 0] *= -1
        theta[np.logical_and(w < 0, theta > 0)] -= np.pi
        theta[np.logical_and(w < 0, theta <= 0)] += np.pi
        t0 = theta / np.pi
        # if density_type == "xy":
        #     mapped_pts = np.zeros_like(vanish_pts[:, :2])
        #     for i in range(vanish_pts.shape[0]):
        #         x, y = vanish_pts[i, :2]
        #         if np.abs(x) > np.abs(y) and np.abs(x) > 0.00001:
        #             r0[i] *= np.sqrt(x**2 + y**2) / np.abs(x)
        #         elif np.abs(x) <= np.abs(y) and np.abs(y) > 0.00001:
        #             r0[i] *= np.sqrt(x**2 + y**2) / np.abs(y)
        #         else:
        #             r0[i] = 0
        #         mapped_pts[i, 0], mapped_pts[i, 1] = r0[i] * np.cos(theta), r0[i] * np.sin(theta)
        # map to int in [0, 63] range.
        mapped_pts = np.stack((r0, t0), axis=1)
        mapped_pts = np.floor((mapped_pts + 1) * np.array([[width/2, height/2]]))
        vanish_hm = generate_gaussian_target(heatmap_shape, mapped_pts, sigma, circular=True)
        return vanish_hm, vanish_pts


def generate_gaussian_target(heatmap_shape, kps, sigma, circular=False):
    h, w = heatmap_shape
    kps = np.floor(kps)
    size = sigma * 3
    lu = np.round(kps - size).astype(np.int16)
    rb = np.round(kps + size).astype(np.int16)
    if circular:
        xs = np.stack((kps[:, 0], kps[:, 0] + w, kps[:, 0] - w), axis=1).reshape(kps.shape[0], 3, 1)
        xdist2 = np.min(np.square(xs - np.arange(w).reshape(1, 1, w)), axis=1).reshape(-1, 1, w)
        ys = np.stack((kps[:, 1], kps[:, 1] + h, kps[:, 1] - h), axis=1).reshape(kps.shape[0], 3, 1)
        ydist2 = np.min(np.square(ys - np.arange(h).reshape(1, 1, h)), axis=1).reshape(-1, h, 1)
        dist2 = xdist2 + ydist2
        gm = np.exp(- dist2 / (2 * sigma ** 2))
        return gm

    else:
        grid = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='xy'), axis=2)
        dis = np.linalg.norm(grid.reshape(1, h, w, 2) - kps.reshape(-1, 1, 1, 2), axis=-1)
        hm = np.exp(-np.square(dis) / (2 * sigma ** 2))
        # hm = np.zeros_like(gm)
        # for i in range(kps.shape[0]):
        #     vis[i] = (lu[i, 0] >= 0) * (lu[i, 1] >= 0) * (rb[i, 0] < w) * (rb[i, 1] < h)
        #     if vis[i]:
        #         hm[lu[i, 1]:rb[i, 1], lu[i, 0]:rb[i, 0]] = gm[lu[i, 1]:rb[i, 1], lu[i, 0]:rb[i, 0]]

        # return hm / np.sum(np.square(hm), axis=(1, 2)).reshape(-1, 1, 1)
        return hm


def gaussian_density_field(heatmap_shape, mus, bones, sigma2):
    """
    mus: n_limbs
    bones: n_limbs x 2 x 2
    sigma2: number
    """
    h, w = heatmap_shape
    grid = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='xy'), axis=2)
    dvs = grid.reshape(1, h, w, 2) - bones[:, 0, :].reshape(-1, 1, 1, 2)
    bvs = bones[:, 1, :] - bones[:, 0, :]
    l = np.linalg.norm(bvs, axis=1)
    bvs = bvs / l.reshape(-1, 1)
    pos = np.sum(dvs * bvs.reshape(-1, 1, 1, 2), axis=-1)

    bns = np.stack((-bvs[:, 1], bvs[:, 0]), axis=1)
    dist2 = np.square(np.sum(dvs * bns.reshape(-1, 1, 1, 2), axis=-1))

    # Set the outside pixels to 0.
    l = l.reshape(-1, 1, 1)
    # pos[pos < 0] = 0
    # pos[pos - l > 0] = 0

    mus = mus.reshape(-1, 1, 1)

    hm_line = mus * l / (mus * l + (1-mus) * pos)**2
    hm_line[hm_line > 1] = 1
    hm = hm_line * np.exp(-dist2/(2*sigma2))
    dist2p1 = np.square(np.linalg.norm(grid.reshape(1, h, w, 2) - bones[:, 0, :].reshape(-1, 1, 1, 2), axis=-1))
    dist2p2 = np.square(np.linalg.norm(grid.reshape(1, h, w, 2) - bones[:, 1, :].reshape(-1, 1, 1, 2), axis=-1))
    p1 = 1 / (mus * l)
    p1[p1 > 1] = 1
    p2 = mus / l
    p2[p2 > 1] = 1
    outerp1 = p1 * np.exp(-dist2p1/(2*sigma2))
    outerp2 = p2 * np.exp(-dist2p2/(2*sigma2))

    hm[pos < 0] = outerp1[pos < 0]
    hm[pos - l > 0] = outerp2[pos - l > 0]

    # return hm / np.sum(hm, axis=(1, 2)).reshape(-1, 1, 1)
    hm = hm / hm.max()
    return hm


def generate_1d_density(size, mus):
    """
    Generate 1 dimensional density map of length size.
    mus: n_limbs
    return n_limbs x size
    """
    x = np.linspace(0, 1, size).reshape(1, size)
    mus = mus.reshape(-1, 1)
    y = mus / (mus + (1 - mus) * x) ** 2
    return y

    
class Human36MMultiViewDataset(Human36MBaseDataset):
    """
    Human3.6M dataset interface, reading monocular data
    """
    def __init__(self, root_dir, label_dir,
                 image_shape=(256, 256),
                 undistort=True,
                 heatmap_shape=(64, 64),
                 sample_level=2,
                 output_type=[],
                 with_damaged_actions=False,
                 transform=None,
                 is_train=False,
                 crop=True,
                 use_gt_data_type=None,
                 use_cameras=[1, 2, 3, 4],
                 stereo_sample=False,
                 sigma=2):
        super(Human36MMultiViewDataset, self).__init__(
            root_dir, label_dir, image_shape, undistort, heatmap_shape, sample_level, output_type,
            with_damaged_actions, transform, is_train, crop
        )

        self.use_gt_data_type = use_gt_data_type
        self.cam_ids = [i-1 for i in use_cameras]
        self.stereo_sample = stereo_sample
        if not is_train:
            missing = []
            for i, label in enumerate(self.labels['table']):
                bbox_hs = np.array([box[2] - box[0] for box in label['bbox_by_camera_tlbr']])
                if np.any(bbox_hs == 0):
                    missing.append(i)
            
            self.labels['table'] = np.delete(self.labels['table'], missing)


    def __len__(self):
        # return 200
        return int(len(self.labels["table"]))

    def __getitem__(self, idx):
        """
        Load data in batches
        """
        sample = defaultdict(list) # return value
        shot = self.labels['table'][idx]

        subject = self.labels['subject_names'][shot['subject_idx']]
        action = self.labels['action_names'][shot['action_idx']]
        frame_idx = shot['frame_idx']
        if self.stereo_sample:
            shuffle(self.cam_ids)
            self.cam_ids = self.cam_ids[:2]

        for camera_idx in self.cam_ids:
            camera_name = self.labels['camera_names'][camera_idx]

            # load bounding box
            bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1,0,3,2]] # TLBR to LTRB
            bbox_height = bbox[2] - bbox[0]

            # load image
            image_path = os.path.join(
                self.root_dir, 'processed', '_undistorted' * self.undistort, subject,
                action, 'imageSequence' + '-undistorted'*self.undistort,
                camera_name, 'img_%06d.jpg' % (frame_idx+1))
            assert os.path.isfile(image_path), f'{image_path} doesn\'t exist, index: {idx:d}' 
            image = cv2.imread(image_path)

            # load camera
            shot_camera = self.labels['cameras'][shot['subject_idx'], camera_idx]
            retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], camera_name)

            if self.crop and bbox_height > 0:
                # crop image
                bbox = normalize_box(bbox)
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
                output.append(action[:-2])
            elif l == "subject":
                output.append(subject)
            # elif l == "lof":
            #     feat_strides = (bbox0[2:] - bbox0[:2]) / np.array(self.heatmap_shape)
            #     kps_in_hm = (joints_2d - bbox0[:2].reshape(1, 2)) / feat_strides.reshape(1, 2)
            #     kps_hm = generate_gaussian_target(self.heatmap_shape, kps_in_hm, self.sigma)
            elif l == "limb_vis":
                output.append(np.ones((len(self.cam_ids), LIMB_PAIRS.shape[0])))
            else:
                output.append(eval(label2value[l]))

        return output
        # elif self.use_gt_data_type == "mu":
        #     mus, density_map, _ = self.generate_gt_density(proj_matrices, keypoints, 2)
        #     return images, proj_matrices, keypoints, mus
        # else:
        #     return images, proj_matrices, keypoints

    def generate_gt_density(self, proj_matrices, keypoints, sigma):
        n_views = proj_matrices.shape[0]
        n_bones = LIMB_PAIRS.shape[0]
        dms = np.zeros((n_views, n_bones, *self.heatmap_shape))
        mus = np.zeros((n_views, n_bones))
        bvs = []
        for i in range(n_views):
            proj = proj_matrices[i]
            kps_2d = project(proj, keypoints)
            depths = np.concatenate((keypoints, np.ones((17, 1))), axis=1) @ proj[2:3, :].T
            mus[i, :] = (depths[LIMB_PAIRS[:, 1]] / depths[LIMB_PAIRS[:, 0]]).flatten()
            bones = kps_2d[LIMB_PAIRS, :]
            bones[:, :, 0] *= self.heatmap_shape[0] / self.image_shape[0]
            bones[:, :, 1] *= self.heatmap_shape[1] / self.image_shape[1]
            dms[i, ...] = gaussian_density_field(self.heatmap_shape, mus[i, :], bones, sigma**2)
            bvs.append(bones)
        
        return mus, dms, np.stack(bvs, axis=0)


def normalize_box(bbox, dst_ratio=1):
    """
    Convert the given bbox to given ratio
    bbox: left, top, right, bottom
    dst_ratio: desired width / height
    """
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    nbbox = np.array(np.round(bbox), dtype=np.int16)
    if w / h > dst_ratio:
        l = int(round(w))
    else:
        l = int(round(h))
    nbbox[0] = np.round((nbbox[0] + nbbox[2] - l) / 2)
    nbbox[2] = nbbox[0] + l
    nbbox[1] = np.round((nbbox[1] + nbbox[3] - l) / 2)
    nbbox[3] = nbbox[1] + l

    return nbbox


def crop_image(img, bbox):
    """
    Crop the image by bbox, pad the image boundary with black.
    """
    h0, w0 = img.shape[:2]
    hb, wb = bbox[3] - bbox[1], bbox[2] - bbox[0]
    if bbox[2] < 0 or bbox[3] < 0 or bbox[0] > w0 or bbox[1] > h0:
        return np.zeros((hb, wb, 3), dtype=np.uint8)
    else:
        padded = np.zeros((h0 + 2*hb, w0 + 2*wb, 3), dtype=np.uint8)
        padded[hb:hb+h0, wb:wb+w0, :] = img

        img = padded[hb+bbox[1]:hb+bbox[3], wb+bbox[0]:wb+bbox[2], :]

        return img
