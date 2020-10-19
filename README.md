# bottom-up-attention.pytorch

This repository contains a **PyTorch** reimplementation of the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) project based on *Caffe*. 

We use [Detectron2](https://github.com/facebookresearch/detectron2) as the backend to provide completed functions including training, testing and feature extraction. Furthermore, we migrate the pre-trained Caffe-based model from the original repository which can extract **the same visual features** as the original model (with deviation < 0.01).

Some example object and attribute predictions for salient image regions are illustrated below. The script to obtain the following visualizations can be found [here](utils/visualize.ipynb)

![example-image](datasets/demo/example_image.jpg?raw=true)

## Table of Contents

0. [Prerequisites](#Prerequisites)
1. [Training](#Training)
2. [Testing](#Testing)
3. [Feature Extraction](#Feature-Extraction)
4. [Pre-trained models](#Pre-trained-models)

## Prerequisites

#### Requirements

- [Python](https://www.python.org/downloads/) >= 3.6
- [PyTorch](http://pytorch.org/) = 1.4
- [Cuda](https://developer.nvidia.com/cuda-toolkit) >= 9.2 and [cuDNN](https://developer.nvidia.com/cudnn)
- [Apex](https://github.com/NVIDIA/apex.git)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Ray](https://github.com/ray-project/ray)
- OpenCV

Note that most of the requirements above are needed for Detectron2. 

#### Installation

1. Clone the project inclduing the required version of Detectron2
   ```bash
   # clone the repository inclduing Detectron2(@5e2a6f6) 
   $ git clone --recursive https://github.com/MILVLG/bottom-up-attention.pytorch
   ```
   
2. Install Detectron2
   ```bash
   $ cd detectron2
   $ pip install -e .
   ```
**Note that the latest version of Detectron2 is incompatible with our project and may result in a running error. Please use the recommended version of Detectron2 (@5e2a6f6) which is downloaded in step 1.** 

3. Compile the rest tools using the following script:

   ```bash
   # install apex
   $ git clone https://github.com/NVIDIA/apex.git
   $ cd apex
   $ python setup.py install
   $ cd ..
   # install the rest modules
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

The following script will train a bottom-up-attention model on the `train` split of VG. *We are still working on this part to reproduce the same results as the Caffe version*. 

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

3. `eval-only` refers to a flag to declare the testing phase.

4. `resume` refers to a flag to declare using the pre-trained model.

## Feature Extraction

Similar with the testing stage, the following script will extract the bottom-up-attention visual features with provided hyper-parameters:

```bash
$ python3 extract_features.py --mode caffe \
         --num-cpus 32 --gpus '0,1,2,3' \
         --extract-mode roi_feats \
         --min-max-boxes '10,100' \
         --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml \ 
         --image-dir <image_dir> --bbox-dir <out_dir> --out-dir <out_dir>  --resume
```

1. `mode = {'caffe', 'detectron2'}` refers to the used mode. For the converted model from Caffe, you need to use the `caffe` mode. For other models trained with Detectron2, you need to use the `detectron2` mode.

2. `num-cpus` refers to the number of cpus to use for ray, and 0 stands for no limit. 

3. `gpus` refers to the ids of gpus to use. 

4. `config-file` refers to all the configurations of the model, which also include the path of the model weights. 

5. `extract-mode` refers to the mode of extract features, including {`roi_feats`, `bboxes` and `bbox_feats`}. 

6. `min-max-boxes` refers to the number of min and max boxes of extractor. 

7. `image-dir` refers to the input image directory.

8. `bbox-dir` refers to the pre-proposed bbox directory.

9. `out-dir` refers to the output feature directory.

10. `resume` refers to a flag to declare using the pre-trained model.

Moreover, using the same pre-trained model, we provide a two-stage strategy for extracting visual features, which results in (slightly) more accurate visual features:

```bash
# extract bboxes only:
$ python3 extract_features.py --mode caffe \
         --num-cpus 32 --gpu '0,1,2,3' \
         --extract-mode bboxes \
         --min-max-boxes '10,100' \
         --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml \ 
         --image-dir <image_dir> --out-dir <out_dir>  --resume 

# extract visual features with the pre-extracted bboxes:
$ python3 extract_features.py --mode caffe \
         --num-cpus 32 --gpu '0,1,2,3' \
         --extract-mode bbox_feats \
         --min-max-boxes '10,100' \
         --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml \ 
         --image-dir <image_dir> --bbox-dir <bbox_dir> --out-dir <out_dir>  --resume 

```


## Pre-trained models

We provided pre-trained models here. The evaluation metrics are exactly the same as those in the original Caffe project. More models will be continuously updated. 

Model | Mode |  Backbone  | Objects mAP@0.5 |Objects weighted mAP@0.5|Download
:-:|:-:|:-:|:-:|:-:|:-:
[Faster R-CNN](./configs/bua-caffe/extract-bua-caffe-r101-fix36.yaml)|Caffe, K=36|ResNet-101|9.3%|14.0%|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EUKhQ3hSRv9JrrW64qpNLSIBGoOjEGCkF8zvgBP9gKax-w?e=kNB9pS)
[Faster R-CNN](./configs/bua-caffe/extract-bua-caffe-r101.yaml)|Caffe, K=[10,100]|ResNet-101|10.2%|15.1%|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EaXvCC3WjtlLvvEfLr3oa8UBLA21tcLh4L8YLbYXl6jgjg?e=SFMoeu)
[Faster R-CNN](./configs/bua-caffe/extract-bua-caffe-r152.yaml)|Caffe, K=100|ResNet-152|11.1%|15.7%|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ETDgy4bY0xpGgsu5tEMzgLcBQjAwpnkKkltNTtPVuMj4GQ?e=rpM1a3)


## License

This project is released under the [Apache 2.0 license](LICENSE).

## Contact

This repo is currently maintained by Jing Li ([@J1mL3e_](https://github.com/JimLee4530)) and Zhou Yu ([@yuzcccc](https://github.com/yuzcccc)).
