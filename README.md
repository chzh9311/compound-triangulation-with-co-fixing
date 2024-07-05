# Compound Triangulation & Co-fixing

This is the official implementation of the TMM paper [*Joint-Limb Compound Triangulation With Co-Fixing for Stereoscopic Human Pose Estimation*](https://ieeexplore.ieee.org/abstract/document/10551483).

*Note: This repository is extended with code for MHAD and joint training.*

## File Structure

* `experiments`: configuration files. The files are all in yaml format.
* `lib`: main code. 
    * `dataset`: the dataloaders. 
    * `models`: network model files.
    * `utils`: tools, functions, data format, etc.
* `backbone_pretrain.py`: file to pretrain 2D backbone before E2E training.
* `config.py`: configuration processors and the default config.
* `main.py`: file to do E2E training.

## How to Use

For convenience, we refer to the root directory of this repo as `${ROOT}`.

### Install Dependencies

First install the latest torch that fits your cuda version, then install the listed requirements. Note: this repository is tested under torch version `1.13.0` and cuda version `11.7`.

```shell
cd ${ROOT}
pip install -r requirement.txt
```

### Prepare Data

**Human3.6M**

* Follow [this guide](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/datasets/human36m_preprocessing/README.md) to prepare image data and labels. Refer to the fetched data directory as `${H36M_ROOT}`, then the directory should look like this:
    ```shell
    ${H36M_ROOT}
        |-- processed
        |     |-- S1/
        |     ...
        |-- extra
        |     |- human36m-multiview-labels-GTbboxes.npy
        |     ...
        ...
    ```
* Generate monocular labels at `${H36M_ROOT}/extra/human36m-monocular-labels-GTbboxes.npy`
    ```shell
    python lib/dataset/convert-multiview-to-monocular.py ${H36M_ROOT}/extra
    ```

**Total Capture**

* Use [Total Capture Toolbox](https://github.com/zhezh/TotalCapture-Toolbox) to prepare data. Suppose the processed data root directory is `${TC_ROOT}` (usually `TotalCapture-Toolbox/data`). It should look like this:

```shell
${TC_ROOT}
    |-- annot
    |     |-- totalcapture_train.pkl
    |     `-- totalcapture_validation.pkl
    `-- images
```

### Pretrained Models

Here we provide the weights for ResNet152 which we used for our model:

* Pretrained on ImageNet: [link](https://download.pytorch.org/models/resnet152-b121ed2d.pth)
* Pretrained on COCO and finetuned on Human3.6M and MPII (From [Learnable Triangulation](https://github.com/karfly/learnable-triangulation-pytorch)): [link](https://disk.yandex.ru/d/hv-uH_7TY0ONpg)

Create a folder named `pretrained` under `${ROOT}` and place the weights in it. If you want the backbone pretraining step to work out-of-the-box, the folder should look like this:

```shell
pretrained
    |-- from_lt/pose_resnet_4.5_pixels_human36m.pth
    `-- pytorch/imagenet/resnet152-b121ed2d.pth
```

We also provide the 4-view weights for Huamn3.6M and Total Capture, which reproduces the results in the paper:

* Human3.6M: [link](https://drive.google.com/file/d/1d-CL9Nlzva_llNwRtELt7lGOy_CGgt8z/view?usp=sharing)
* Total Capture: [link](https://drive.google.com/file/d/1sjvx5d7woQKDPkOLiZpoFgRW-9JwFTdU/view?usp=sharing)

Place the weights at a certain directory so the path could be referred to as `${weight_path}`. We will be using this path in the **Testing** stage.

### Training

We train the model in a two-step manner: first train the 2D backbone which outputs the joint confidence heatmap and the LOF. Then we train the model end-to-end for better accuracy.

To do pretraining, just run:

```shell
python backbone_pretrain.py --cfg experiments/ResNet${n_layers}/${dataset}-${resolution}-backbone.yaml --runMode train
```

Then, to train the model end-to-end:

```shell
python main.py --cfg experiments/ResNet${n_layers}/${dataset}-${resolution}.yaml --runMode train
```

### Testing

```shell
python main.py\
     --cfg experiments/ResNet${n_layers}/${dataset}-${resolution}.yaml\
     --runMode test\
     -w ${weight_path}
```

For this repo, we provide `n_layers=152`, `dataset=human3.6m | totalcapture` and `resolution=384x384 | 320x320` as an example.

*Note: If you wish to train or test using multiple GPUS, please specify the GPU ids in the config file. By default, the script only uses GPU 0 for training / testing.*

## Citation

If you use our code, please cite us with:

```
@article{zhuo2024compound,
  author={Chen, Zhuo and Wan, Xiaoyue and Bao, Yiming and Zhao, Xu},
  journal={IEEE Transactions on Multimedia}, 
  title={Joint-Limb Compound Triangulation With Co-Fixing for Stereoscopic Human Pose Estimation}, 
  year={2024},
  pages={1-11},
  doi={10.1109/TMM.2024.3410514}}
```