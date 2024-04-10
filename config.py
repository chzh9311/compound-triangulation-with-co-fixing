import os
import yaml
import logging
import json
from easydict import EasyDict as edict


def get_config(name):
    try:
        with open(name, encoding="utf-8") as f:
            cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
        return cfg
    except FileNotFoundError as e:
        logging.log(level=logging.ERROR, msg="File %s Not found. Make sure you have the right name typed in." % name)
        exit(-1)
    

def save_config(config, path):
    with open(path, "w", encoding="utf-8") as wf:
        yaml.dump(config, wf, indent=4, default_flow_style=False)


def overwrite_cfg(config0, config1):
    """
    overwrite relative configurations in config0 with config1
    0 and 1 are dictioinaries.
    """
    for k, v in config0.items():
        if k in config1:
            if type(v) is edict:
                config0[k] = overwrite_cfg(config0[k], config1[k])
            else:
                config0[k] = config1[k]
        else:
            config0[k] = config1[k]
    return config0


def update_config(config, args):
    if args.cfg:
        new_cfg = get_config(args.cfg)
        config = overwrite_cfg(config, new_cfg)
    if args.runMode == "train" and args.dir:
        config.MODEL.BACKBONE_WEIGHTS = os.path.join(args.dir, "weights", "best.pth")
    elif args.runMode == "train" and args.epochs:
        config.TRAIN.NUM_EPOCHS = args.epochs
    return config

config = edict()

config = edict()
config.GPUS = [0, 1]

config.DATASET = edict()
config.DATASET.NAME = "totalcapture"
config.DATASET.TC_ROOT = "/home/chenzhuo/data/totalcapture/TotalCapture-Toolbox/data/images"
config.DATASET.TC_LABELS = "/home/chenzhuo/data/totalcapture/TotalCapture-Toolbox/data/annot"
config.DATASET.H36M_ROOT = "/home/chenzhuo/data/h36m-fetch/processed"
config.DATASET.H36M_LABELS = "/home/chenzhuo/data/h36m-fetch/extra/human36m-multiview-labels-GTbboxes.npy"
config.DATASET.H36M_MONOLABELS = "data/h36m/labels/human36m-monocular-labels-GTbboxes.npy"
config.DATASET.WITH_DAMAGED_ACTIONS = True

config.MODEL = edict()
config.MODEL.NAME = "lofpose"
config.MODEL.STYLE = "pytorch"
config.MODEL.SOFTMAX_BETA = 100
config.MODEL.IMAGE_SIZE = [320, 320]
config.MODEL.NUM_JOINTS = 16
config.MODEL.NUM_LIMBS = 15
config.MODEL.INIT_WEIGHTS = True
config.MODEL.LOAD_FINAL_WEIGHTS = True
config.MODEL.LOAD_DECONVS = True
config.MODEL.PRETRAINED = ""
config.MODEL.BACKBONE_WEIGHTS = "log/end2end/train_ttc_cam13_pure2d/epoch0-2/weights/best.pth"
config.MODEL.LAMBDA = 0
config.MODEL.BACKBONE = "ResNet"
config.MODEL.USE_CONFIDENCE = True
config.MODEL.NUM_DIMS = 3
config.MODEL.USE_LOF = False
config.MODEL.FUSION_STRATEGY = None
config.MODEL.CONFIDENCE_NORM = True

config.MODEL.CO_FIXING = edict()
config.MODEL.CO_FIXING.FIX_HEATMAP = False
config.MODEL.CO_FIXING.FIX_ALPHA = 2
config.MODEL.CO_FIXING.VEC_IMPROVE_TH = 0.25
config.MODEL.CO_FIXING.VEC_FIX_UB = 0.1
config.MODEL.CO_FIXING.PTS_IMPROVE_TH = 0.25
config.MODEL.CO_FIXING.PTS_FIX_UB = 400
config.MODEL.REQUIRED_DATA = ['images', 'keypoints3d', 'projections', 'intrinsics', 'cam_ctr', 'rotation', 'identity', 'index', 'subject']
config.MODEL.MODEL_OUTPUT = ['keypoints3d', 'keypoints2d', 'heatmap', 'confidences']
config.MODEL.BACKBONE_OUTPUT = ['heatmap', 'confidences']

config.MODEL.EXTRA = edict()
config.MODEL.EXTRA.SIGMA = 2
config.MODEL.EXTRA.HEATMAP_SIZE = [80, 80]
config.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
config.MODEL.EXTRA.DECONV_WITH_BIAS = False
config.MODEL.EXTRA.NUM_DECONV_LAYERS = 3
config.MODEL.EXTRA.NUM_DECONV_FILTERS = [256, 256, 256]
config.MODEL.EXTRA.NUM_DECONV_KERNELS = [4, 4, 4]
config.MODEL.EXTRA.NUM_LAYERS = 152

config.TRAIN = edict()
config.TRAIN.IS_PRETRAIN = True
config.TRAIN.BATCH_SIZE = 8
config.TRAIN.USE_CAMERAS = [1, 3]
config.TRAIN.REFINE_INDICATOR = 0
config.TRAIN.DATA_PARALLEL = True
config.TRAIN.LEARNING_RATE = 0.0001

config.TRAIN.SCHEDULER = edict()
config.TRAIN.SCHEDULER.STEP = 10
config.TRAIN.SCHEDULER.GAMMA = 0.1
config.TRAIN.SHUFFLE = True
config.TRAIN.SOFT_EP = 400
config.TRAIN.SVD_EPS = 0.01
config.TRAIN.LOSS_WEIGHT = [10000.0, 0]
config.TRAIN.NUM_WORKERS = 8
config.TRAIN.NUM_EPOCHS = 10
config.TRAIN.CONTINUE = False
config.TRAIN.CHECKPOINT = "log/end2end/train_ttc_cam14_pure2d/weights/checkpoint_epoch4_loss4.166067e+01.pkl"
config.TRAIN.MID_CHECKPOINTS = 0
config.TRAIN.LOSS_FREQ = 20
config.TRAIN.VIS_FREQ = 2000

config.TEST = edict()
config.TEST.WRITE_LOG = True
config.TEST.BATCH_SIZE = 32
config.TEST.DATA_PARALLEL = True
config.TEST.SCA_STEPS = 3
config.TEST.USE_GT_DATA_TYPE = None
config.TEST.LAMBDA = 0
config.TEST.VIS_FREQ = 1
config.TEST.SHUFFLE = False
config.TEST.NUM_WORKERS = 8
config.TEST.USE_CAMERAS = [1, 3]
config.TEST.FRAME_SAMPLE_RATE = 64


if __name__ == "__main__":
    cfg = get_config()
    save_config(cfg, "test.yaml")
