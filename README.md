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
- [PyTorch](http://pytorch.org/) >= 1.4
- [Cuda](https://developer.nvidia.com/cuda-toolkit) >= 9.2 and [cuDNN](https://developer.nvidia.com/cudnn)
- [Apex](https://github.com/NVIDIA/apex.git)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Ray](https://github.com/ray-project/ray)
- [OpenCV](https://opencv.org/)
- [Pycocotools](https://github.com/cocodataset/cocoapi)

Note that most of the requirements above are needed for Detectron2. 

#### Installation

1. Clone the project including the required version (v0.2.1) of Detectron2
   ```bash
   # clone the repository inclduing Detectron2(@be792b9) 
   $ git clone --recursive https://github.com/MILVLG/bottom-up-attention.pytorch
   ```
   
2. Install Detectron2
   ```bash
   $ cd detectron2
   $ pip install -e .
   $ cd ..
   ```
**We recommend using Detectron2 v0.2.1 (@be792b9) as backend for this project, which has been cloned in step 1. We believe a newer Detectron2 version is also compatible with this project unless their interface has been changed (we have tested v0.3 with PyTorch 1.5).**
   
3. Compile the rest tools using the following script:

   ```bash
   # install apex
   $ git clone https://github.com/NVIDIA/apex.git
   $ cd apex
   $ python setup.py install
   $ cd ..
   # install the rest modules
   $ python setup.py build develop
   $ pip install ray
   ```

#### Setup

If you want to train or test the model, you need to download the images and annotation files of the Visual Genome (VG) dataset. **If you only need to extract visual features using the pre-trained model, you can skip this part**.

The original VG images ([part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)) are to be downloaded and unzipped to the `datasets` folder.

The generated annotation files in the original repository are needed to be transformed to a COCO data format required by Detectron2. The preprocessed annotation files can be downloaded [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EWpiE_5PvBdKiKfCi0pBx_EB5ONo8D8XABUz7tWcnltCrw?e=xIeW23) and unzipped to the `dataset` folder.

Finally, the `datasets` folders will have the following structure:

```angular2html
|-- datasets
   |-- vg
   |  |-- images
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
         --eval-only
```

1. `mode = {'caffe', 'detectron2'}` refers to the used mode. For the converted model from Caffe, you need to use the `caffe` mode. For other models trained with Detectron2, you need to use the `detectron2` mode.

2. `config-file` refers to all the configurations of the model, which also include the path of the model weights. 

3. `eval-only` refers to a flag to declare the testing phase.

## Feature Extraction

With highly-optimized multi-process parallelism, the following script will extract the bottom-up-attention visual features in a fast manner (about 7 imgs/s on a workstation with 4 Titan-V GPUs and 32 CPU cores). 

And we also provide a [faster version](extract_features_faster.py) of the script of extract features, which will extract the bottom-up-attention visual features in **an extremely fast manner!** (about 16 imgs/s on a workstation with 4 Titan-V GPUs and 32 cores) However, it has a drawback that it could cause memory leakage problem when the computing capability of GPUs and CPUs is mismatched (More details and some matched examples in [here](https://github.com/MILVLG/bottom-up-attention.pytorch/pull/41)). 

To use this faster version, just replace 'extract_features.py' with 'extract_features_faster.py' in the following script. **MAKE SURE YOU HAVE ENOUGH CPUS.**

```bash
$ python3 extract_features.py --mode caffe \
         --num-cpus 32 --gpus '0,1,2,3' \
         --extract-mode roi_feats \
         --min-max-boxes '10,100' \
         --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml \ 
         --image-dir <image_dir> --bbox-dir <out_dir> --out-dir <out_dir>
```

1. `mode = {'caffe', 'detectron2'}` refers to the used mode. For the converted model from Caffe, you need to use the `caffe` mode. For other models trained with Detectron2, you need to use the `detectron2` mode. `'caffe'` is the default value.

2. `num-cpus` refers to the number of cpu cores to use for accelerating the cpu computation. **0** stands for using all possible cpus and **1** is the default value. 

3. `gpus` refers to the ids of gpus to use. **'0'** is the default value.

4. `config-file` refers to all the configurations of the model, which also include the path of the model weights. 

5. `extract-mode` refers to the modes for feature extraction, including {`roi_feats`, `bboxes` and `bbox_feats`}. 

6. `min-max-boxes` refers to the min-and-max number of features (boxes) to be extracted. 

7. `image-dir` refers to the input image directory.

8. `bbox-dir` refers to the pre-proposed bbox directory. Only be used if the `extract-mode` is set to `'bbox_feats'`.

9. `out-dir` refers to the output feature directory.

Using the same pre-trained model, we provide an alternative *two-stage* strategy for extracting visual features, which results in (slightly) more accurate bboxes and visual features:

```bash
# extract bboxes only:
$ python3 extract_features.py --mode caffe \
         --num-cpus 32 --gpu '0,1,2,3' \
         --extract-mode bboxes \
         --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml \ 
         --image-dir <image_dir> --out-dir <out_dir>  --resume 

# extract visual features with the pre-extracted bboxes:
$ python3 extract_features.py --mode caffe \
         --num-cpus 32 --gpu '0,1,2,3' \
         --extract-mode bbox_feats \
         --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml \ 
         --image-dir <image_dir> --bbox-dir <bbox_dir> --out-dir <out_dir>  --resume 

```

## Pre-trained models

We provided pre-trained models as follows, including the models converted from the original [Caffe repo](https://github.com/peteanderson80/bottom-up-attention) (the standard [dynamic 10-100 model](https://www.dropbox.com/s/5xethd2nxa8qrnq/resnet101_faster_rcnn_final.caffemodel?dl=1) and the alternative [fix36 model](https://www.dropbox.com/s/2h4hmgcvpaewizu/resnet101_faster_rcnn_final_iter_320000.caffemodel?dl=1)). The evaluation metrics are exactly the same as those in the original Caffe project. 

Model | Mode |  Backbone  | Objects mAP@0.5 |Objects weighted mAP@0.5|Download
:-:|:-:|:-:|:-:|:-:|:-:
[Faster R-CNN-k36](./configs/bua-caffe/extract-bua-caffe-r101-fix36.yaml)|Caffe|ResNet-101|9.3%|14.0%|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EUKhQ3hSRv9JrrW64qpNLSIBGoOjEGCkF8zvgBP9gKax-w?download=1)
[Faster R-CNN-k10-100](./configs/bua-caffe/extract-bua-caffe-r101.yaml)|Caffe|ResNet-101|10.2%|15.1%|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EaXvCC3WjtlLvvEfLr3oa8UBLA21tcLh4L8YLbYXl6jgjg?download=1)
[Faster R-CNN](./configs/bua-caffe/extract-bua-caffe-r152.yaml)|Caffe|ResNet-152|11.1%|15.7%|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ETDgy4bY0xpGgsu5tEMzgLcBQjAwpnkKkltNTtPVuMj4GQ?download=1)


## License

This project is released under the [Apache 2.0 license](LICENSE).

## Contact

This repo is currently maintained by Zhou Yu ([@yuzcccc](https://github.com/yuzcccc)), Tongan Luo ([@Zoroaster97](https://github.com/Zoroaster97)), and Jing Li ([@J1mL3e_](https://github.com/JimLee4530)).

## Citation

If this repository is helpful for your research or you want to refer the provided pretrained models, you could cite the work using the following BibTeX entry:

```
@misc{yu2020buapt,
  author = {Yu, Zhou and Li, Jing and Luo, Tongan and Yu, Jun},
  title = {A PyTorch Implementation of Bottom-Up-Attention},
  howpublished = {\url{https://github.com/MILVLG/bottom-up-attention.pytorch}},
  year = {2020}
}

```