from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from collections import OrderedDict

from lib.utils.functions import fit_mu_and_var
from lib.utils.utils import make_logger
from lib.models.layers import GlobalAveragePoolingHead, CoFixing
from lib.utils.DictTree import create_human_tree

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FieldPoseNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.load_final_weights = cfg.MODEL.LOAD_FINAL_WEIGHTS
        self.use_conf = cfg.MODEL.USE_CONFIDENCE

        super(FieldPoseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.encode_planes = self.inplanes
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        self.use_lof = cfg.MODEL.USE_LOF
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.num_limbs = cfg.MODEL.NUM_LIMBS
        if self.use_lof:
            # Restore settings.
            self.inplanes = self.encode_planes
            self.softmax_beta = cfg.MODEL.SOFTMAX_BETA
            self.deconv_layers2 = self._make_deconv_layer(
                extra.NUM_DECONV_LAYERS,
                extra.NUM_DECONV_FILTERS,
                extra.NUM_DECONV_KERNELS,
            )

            self.final_layer2 = nn.Conv2d(
                in_channels=extra.NUM_DECONV_FILTERS[-1],
                out_channels=cfg.MODEL.NUM_LIMBS * cfg.MODEL.NUM_DIMS,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            )
            if self.use_conf:
                self.confidences = GlobalAveragePoolingHead(512*block.expansion, cfg.MODEL.NUM_JOINTS+cfg.MODEL.NUM_LIMBS)
        elif self.use_conf:
            # self.vol_confidences = GlobalAveragePoolingHead(512*block.expansion, 32)
            self.alg_confidences = GlobalAveragePoolingHead(512*block.expansion, cfg.MODEL.NUM_JOINTS)
        
        # self.is_pretrain=cfg.TRAIN.IS_PRETRAIN
        # self.NUM_JOINTS = cfg.MODEL.NUM_JOINTS

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, conv_dim=2):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                eval(f"nn.ConvTranspose{conv_dim}d")(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias if conv_dim == 2 else True))
            layers.append(eval(f"nn.BatchNorm{conv_dim}d")(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.use_lof:
            x1 = self.deconv_layers(x)
            x1 = self.final_layer(x1)
            x2 = self.deconv_layers2(x)
            x2 = self.final_layer2(x2)
            if self.use_conf:
                conf = self.confidences(x) # 16 + 17
                return x1, x2, conf
            else:
                return x1, x2
        else:
            features = self.deconv_layers(x)
            x1 = self.final_layer(features)
            if self.use_conf and hasattr(self, 'alg_confidences'):
                conf = self.alg_confidences(x)
                return x1, conf
            else:
                return x1

    def init_weights(self, pretrained='', load_pretrained_deconv=True):
            # pretrained_state_dict = torch.load(pretrained)
        #     logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
        assert os.path.exists(pretrained), f"No such file or dir: {pretrained}"
        if pretrained[-4:] == ".pkl":
            with open(pretrained, "rb") as ckpf:
                ckp = pickle.load(ckpf)
                state_dict_old = ckp['model']
        else:
            state_dict_old = torch.load(pretrained)
        # if isinstance(checkpoint, OrderedDict):
        #     state_dict_old = checkpoint
        # elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        #     state_dict_old = checkpoint['state_dict']
        # else:
        #     raise RuntimeError(
        #         'No state_dict found in checkpoint file {}'.format(pretrained))
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            if key.startswith('module.'):
                # state_dict[key[7:]] = state_dict[key]
                # state_dict.pop(key)
                state_dict[key[7:]] = state_dict_old[key]
            else:
                state_dict[key] = state_dict_old[key]

        if not load_pretrained_deconv:
            for name, m in self.deconv_layers.named_modules():
                # del(state_dict[name+".weight"])
                # del(state_dict[name+".bias"])
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

        if hasattr(self, "deconv_layers2"):
            for name, m in self.deconv_layers2.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer2.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

        # Remove the parameters of final_layer in pretrained model
        if not self.load_final_weights:
            del (state_dict["final_layer.weight"])
            del (state_dict["final_layer.bias"])
        else:
            state_dict["final_layer.weight"] = state_dict["final_layer.weight"][:17, ...]
            state_dict["final_layer.bias"] = state_dict["final_layer.bias"][:17]
            if hasattr(self, "deconv_layers2"):
                state_dict["final_layer2.weight"] = state_dict["final_layer2.weight"][:48, ...]
                state_dict["final_layer2.bias"] = state_dict["final_layer2.bias"][:48]
        self.load_state_dict(state_dict, strict=False)
        # else:
        #     logger.error('=> imagenet pretrained model dose not exist')
        #     logger.error('=> please download it first')
        #     raise ValueError('imagenet pretrained model does not exist')

    def load_backbone_params(self, weight_path, load_confidences=False):
        with open(weight_path, "rb") as wf:
            if weight_path[-4:] == ".pkl":
                checkpoint = pickle.load(wf)
                state_dict = checkpoint["model"]
            else:
                state_dict = torch.load(wf)
        new_state_dict = OrderedDict()

        # handle with the modules. introduced in DataParallel
        for key in state_dict.keys():
            if key.startswith("module.backbone."):
                new_state_dict[key[16:]] = state_dict[key]
            elif key.startswith("module."):
                new_state_dict[key[7:]] = state_dict[key]
            elif key.startswith("backbone."):
                new_state_dict[key[9:]] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
            
        # for key in new_state_dict.keys():
        #     if key.startswith("final_layer2"):
        #         new_state_dict[key] = new_state_dict[key][:48]
        #     elif key.startswith("final_layer"):
        #         new_state_dict[key] = new_state_dict[key][:17]
        #     elif key.startswith("confidences.head.4"):
        #         new_state_dict[key] = new_state_dict[key][:33]
            
        self.load_state_dict(new_state_dict, strict=False)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}
            

def get_FPNet(cfg, is_train=False, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    style = cfg.MODEL.STYLE

    block_class, layers = resnet_spec[num_layers]

    if style == 'caffe':
        block_class = Bottleneck_CAFFE

    model = FieldPoseNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, cfg.MODEL.LOAD_DECONVS)
        print(f"Pretrained weights loaded from {cfg.MODEL.PRETRAINED}")

    return model


if __name__ == "__main__":
    from config import get_config
    model = get_FPNet(get_config())
