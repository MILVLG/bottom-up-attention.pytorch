# bottom-up-attention.pytorch

This repository contains a **PyTorch** reimplementation of the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) project based on *Caffe*. 

We use [Detectron2](https://github.com/facebookresearch/detectron2) as the backend to provide completed functions including training, testing and feature extraction. Furthermore, we migrate the pre-trained Caffe-based model from the original repository which obtains **the same visual features** as the original model (with deviation < 0.01). To the best of our knowledge, we are the first success attempt to migrate the pre-trained Caffe model. 

## Table of Contents

0. [Prerequisites](#Prerequisites)
1. [Training](#Training)
2. [Testing](#Testing)
3. [Feature Extraction](#Feature-Extraction)
4. [Pre-trained models](#Pre-trained-models)

## Prerequisites

#### Requirements

- [Python](https://www.python.org/downloads/) >= 3.6
- [PyTorch](http://pytorch.org/) >= 1.4
- [Cuda](https://developer.nvidia.com/cuda-toolkit) >= 9.2 and [cuDNN](https://developer.nvidia.com/cudnn)
- [Apex](https://github.com/NVIDIA/apex.git)
- [Detectron2](https://github.com/facebookresearch/detectron2)

Note that most of the requirements above are needed for Detectron2. 

#### Installation

1. Install Detectron2 according to their official instructions [here](https://github.com/facebookresearch/detectron2/blob/5e2a6f62ef752c8b8c700d2e58405e4bede3ddbe/INSTALL.md).

2. Compile other used tools using the following script:

   ```bash
   # install apex
   $ git clone https://github.com/NVIDIA/apex.git
   $ cd apex
   $ python setup.py install
   $ cd ..
   $ python setup.py build develop
   ```

#### Setup

If you want to train or test the model, you need to download the images and annotation files of the Visual Genome (VG) dataset. **If you only need to extract visual features using the pre-trained model, you can skip this part**.

The original VG images ([part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)) are to be downloaded and unzipped to the `datasets` folder.

The generated annotation files in the original repository are needed to be transformed to a COCO data format required by Detectron2. The preprocessed annotation files can be downloaded [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EWpiE_5PvBdKiKfCi0pBx_EB5ONo8D8XABUz7tWcnltCrw?e=xIeW23) and unzipped to the `dataset` folder.

Finally, the `datasets` folders will have the following structure:

```angular2html
|-- datasets
   |-- vg
   |  |-- image
   |  |  |-- VG_100K
   |  |  |  |-- 2.jpg
   |  |  |  |-- ...
   |  |  |-- VG_100K_2
   |  |  |  |-- 1.jpg
   |  |  |  |-- ...
   |  |-- annotations
   |  |  |-- train.json
   |  |  |-- val.json
```

## Training

The following script will train a bottom-up-attention model on the `train` split of VG:

```bash
$ python3 train_net.py --mode detectron2 \
         --config-file configs/bua-caffe/train-bua-caffe-r101.yaml \ 
         --resume
```

1. `mode = {'caffe', 'detectron2'}` refers to the used mode. We only support the mode with Detectron2, which refers to `detectron2` mode, since we think it is unnecessary to train a new model using the `caffe` mode. 

2. `config-file` refers to all the configurations of the model.

3. `resume` refers to a flag if you want to resume training from a specific checkpoint.

## Testing

Given the trained model, the following script will test the performance on the `val` split of VG:

```bash
$ python3 train_net.py --mode caffe \
         --config-file configs/bua-caffe/test-bua-caffe-r101.yaml \ 
         --eval-only --resume
```

1. `mode = {'caffe', 'detectron2'}` refers to the used mode. For the converted model from Caffe, you need to use the `caffe` mode. For other models trained with Detectron2, you need to use the `detectron2` mode.

2. `config-file` refers to all the configurations of the model, which also include the path of the model weights. 

3. `eval-only` refers to a flag if you want to testing only.

4. `resume` refers to a flag if you want to resume training from a specific checkpoint.

## Feature Extraction

Similar with the testing stage, the following script will extract the bottom-up-attention visual features with provided hyper-parameters:

```bash
$ python3 extract_feature.py --mode caffe \
         --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml \ 
         --image-dir <image_dir> --out-dir <out_dir> --resume
```

1. `mode = {'caffe', 'detectron2'}` refers to the used mode. For the converted model from Caffe, you need to use the `caffe` mode. For other models trained with Detectron2, you need to use the `detectron2` mode.

2. `config-file` refers to all the configurations of the model, which also include the path of the model weights. 

3. `image-dir` refers to the input image directory.

4. `out-dir` refers to the output feature directory.

5. `resume` refers to a flag if you want to resume training from a specific checkpoint.

## Pre-trained models

We provided pre-trained models here. The evaluation metrics are exactly the same as those in the original Caffe project.

Currently we only provide the converted model from Caffe, which report exactly the same scores compared to the original version. More models will be continuously updated. 

Model  |  Backbone  | Objects mAP@0.5 |Objects weighted mAP@0.5|Download
:-:|:-:|:-:|:-:|:-:
Faster R-CNN （Caffe）|ResNet-101|10.2%|15.1%|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EaORTX7eenZOgGDjKe03e6UB31ty7Q2bkAN-LEKrqjSa6A?e=6iQGAj)

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Contact

This repo is currently maintained by Jing Li ([@J1mL3e_](https://github.com/JimLee4530)) and Zhou Yu ([@yuzcccc](https://github.com/yuzcccc)).