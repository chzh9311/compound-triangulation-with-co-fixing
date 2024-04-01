import sys
import os

import numpy as np
from tqdm import tqdm

from Camera import Camera

sys.path.append("/home/chenzhuo/Projects/DNN_3DHPE/Mine/DensityFieldPose/lib/utils")
from functions import project


label_path = sys.argv[1]

with open(label_path, 'rb') as lf:
    labels = np.load(lf, allow_pickle=True).item()

subjects = labels['subject_names']
actions = labels['action_names']
cameras = labels["camera_names"]

table_dtype = np.dtype([
    ('subject_idx', np.int8),
    ('action_idx', np.int8),
    ('frame_idx', np.int16),
    ('camera_idx', np.int16),
    ('keypoints_2d', np.float32, (17,2)), # roughly MPII format
    ('keypoints_3d', np.float32, (17,3)), # roughly MPII format
    ('bbox_by_camera_tlbr', np.int16, (4,))
])   

cam_projs = {}
for s in range(len(subjects)):
    for c in range(len(cameras)):
        params = labels["cameras"][s, c]
        cam = Camera(params['R'], params['t'], params['K'], params['dist'], cameras[c])
        cam_projs[(s, c)] = cam.projection

all_info = []
for frame in tqdm(labels["table"]):
    image_info = np.empty((1,), dtype=table_dtype)
    for c in range(len(cameras)):
        s = frame["subject_idx"]
        image_info["subject_idx"] = s
        image_info["action_idx"] = frame["action_idx"]
        image_info["frame_idx"] = frame["frame_idx"]
        image_info["camera_idx"] = c
        pose_3d = frame["keypoints"]
        kps_2d = project(cam_projs[(s, c)], pose_3d)
        image_info["keypoints_2d"] = kps_2d
        image_info["keypoints_3d"] = pose_3d
        bbox = image_info['bbox_by_camera_tlbr'] = frame['bbox_by_camera_tlbr'][c, :]
        if bbox[2] - bbox[0] == 0:
            print("Invalid value excluded.")
            continue
        all_info.append(image_info)

monocular_info = np.concatenate(all_info)
labels["table"] = monocular_info

np.save("../../data/h36m/labels/human36m-monocular-labels-GTbboxes.npy", labels)