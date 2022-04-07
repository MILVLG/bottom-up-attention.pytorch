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

1. Clone the project including the required version (v0.2.1) of Detectron2. **Note that if you use another version, some strange problems may occur**.
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

The original VG images ([part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)) are to be downloaded and unzipped to one folder and put it into the `datasets` folder.

The generated annotation files in the original repository are needed to be transformed to a COCO data format required by Detectron2. The preprocessed annotation files can be downloaded [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EWpiE_5PvBdKiKfCi0pBx_EB5ONo8D8XABUz7tWcnltCrw?download=1) and unzipped to the `dataset` folder.

Finally, the `datasets` folders will have the following structure:

```angular2html
|-- datasets
   |-- visual_genome
   |  |-- images
   |  |  |  |-- 1.jpg
   |  |  |  |-- 2.jpg
   |  |  |  |-- ...
   |  |  |  |-- ...
   |  |-- annotations
   |  |  |-- visual_genome_train.json
   |  |  |-- visual_genome_test.json
   |  |  |-- visual_genome_val.json
```

## Training

The following script will train a bottom-up-attention model on the `train` split of VG. 

```bash
$ python3 train_net.py --mode d2 \
         --config-file configs/d2/train-d2-r101.yaml \
         --resume
```

1. `mode = 'd2'` refers to training a model with the Detectron2 backend, which is inspired by the [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa/). We think it is unnecessary to train a new model using the `caffe` mode. The pre-trained Caffe models are provided for testing and feature extraction. 

2. `config-file` refers to all the configurations of the model.

3. `resume` refers to a flag if you want to resume training from a specific checkpoint. 

## Testing

Given the trained model, the following script will test the performance on the `val` split of VG:

```bash
$ python3 train_net.py --mode caffe \
         --config-file configs/caffe/test-caffe-r101.yaml \
         --eval-only
```

1. `mode = {'caffe', 'd2'}` refers to the used mode. For the converted model from Caffe, you need to use the `caffe` mode. For other models trained with Detectron2, you need to use the `d2` mode.

2. `config-file` refers to all the configurations of the model, which also include the path of the model weights. 

3. `eval-only` refers to a flag to declare the testing phase.

## Feature Extraction

Given the trained model, the following script will extract the bottom-up-attention visual features. Single GPU and multiple GPUs are both supported. 

```bash
$ python3 extract_features.py --mode caffe \
         --num-cpus 32 --gpus '0,1,2,3' \
         --extract-mode roi_feats \
         --min-max-boxes '10,100' \
         --config-file configs/caffe/test-caffe-r101.yaml \
         --image-dir <image_dir> --bbox-dir <out_dir> --out-dir <out_dir>
         --fastmode
```

1. `mode = {'caffe', 'd2'}` refers to the used mode. For the converted model from Caffe, you need to use the `caffe` mode. For other models trained with Detectron2, you need to use the `detectron2` mode. `'caffe'` is the default value. **Note** that the `detecron2` mode need to run with [Ray](https://github.com/ray-project/ray).

2. `num-cpus` refers to the number of cpu cores to use for accelerating the cpu computation. **0** stands for using all possible cpus and **1** is the default value. 

3. `gpus` refers to the ids of gpus to use. **'0'** is the default value. If the number of gpus greater than 1, for example **'0,1,2,3'**, the script will use the [Ray](https://github.com/ray-project/ray) library for parallelization.

4. `config-file` refers to all the configurations of the model, which also include the path of the model weights. 

5. `extract-mode` refers to the modes for feature extraction, including {`roi_feats`, `bboxes` and `bbox_feats`}. 

6. `min-max-boxes` refers to the min-and-max number of features (boxes) to be extracted. **Note**  that `mode d2` only support to set the min-and-max number as `'100,100'` to get 100 boxes per image or other values to get about 50~60 boxes per image.

7. `image-dir` refers to the input image directory.

8. `bbox-dir` refers to the pre-proposed bbox directory. Only be used if the `extract-mode` is set to `'bbox_feats'`.

9. `out-dir` refers to the output feature directory.

10. `fastmode` refers to use the a faster version (about `2x` faster on a workstation with 4 Titan-V GPUs and 32 CPU cores), at the expense of a potential memory leakage problem if the computing capability of GPUs and CPUs is mismatched. More details and some matched examples in [here](https://github.com/MILVLG/bottom-up-attention.pytorch/pull/41).

    

Using the same pre-trained model, we also provide an alternative *two-stage* strategy for extracting visual features. This results in (slightly) more accurate bounding boxes and visual features, at the expense of more time overhead:

```bash
# extract bboxes only:
$ python3 extract_features.py --mode caffe \
         --num-cpus 32 --gpu '0,1,2,3' \
         --extract-mode bboxes \
         --config-file configs/caffe/test-caffe-r101.yaml \
         --image-dir <image_dir> --out-dir <out_dir>  --resume 

# extract visual features with the pre-extracted bboxes:
$ python3 extract_features.py --mode caffe \
         --num-cpus 32 --gpu '0,1,2,3' \
         --extract-mode bbox_feats \
         --config-file configs/caffe/test-caffe-r101.yaml \
         --image-dir <image_dir> --bbox-dir <bbox_dir> --out-dir <out_dir>  --resume 

```

## Pre-trained models

We provided pre-trained models as follows, including the models trained in both the `caffe` and `d2` mode. 

For the models of the `caffe` mode, `R101-k36` and `R101-k10-100` refer to the [fix36 model](https://www.dropbox.com/s/2h4hmgcvpaewizu/resnet101_faster_rcnn_final_iter_320000.caffemodel?dl=1) and [dynamic 10-100 model](https://www.dropbox.com/s/5xethd2nxa8qrnq/resnet101_faster_rcnn_final.caffemodel?dl=1) provided in the original [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) repository. We additionally provide a `R-152` model which outperforms the two counterparts above.  

For the models of the `d2` mode, we follow the configurations and implementations in the [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa/) and trained three models using the training script in this repository, namely `R50`, `R101` and `X152`.

name | mode | objects mAP@0.5 |weighted objects mAP@0.5|download
:-:|:-:|:-:|:-:|:-:
[R101-k36](./configs/caffe/test-caffe-r101-fix36.yaml)|caffe|9.3|14.0|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EUKhQ3hSRv9JrrW64qpNLSIBGoOjEGCkF8zvgBP9gKax-w?download=1)
[R101-k10-100](./configs/caffe/test-caffe-r101.yaml)|caffe|10.2|15.1|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EaXvCC3WjtlLvvEfLr3oa8UBLA21tcLh4L8YLbYXl6jgjg?download=1)
[R152](./configs/caffe/test-caffe-r152.yaml)|caffe|**11.1**|15.7|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ETDgy4bY0xpGgsu5tEMzgLcBQjAwpnkKkltNTtPVuMj4GQ?download=1)
[R50](./configs/d2/test-d2-r50.yaml)|d2|8.2|14.9|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EfYoinBHrFlKmKonocse8yEBXN-hyCHNygYqjxGpIBsPvQ?download=1)
[R101](./configs/d2/test-d2-r101.yaml)|d2|9.2|15.9|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EXXItFlOpHlNq81O1H_cPyoBXUPyXoHmIwPEudnTWKX4rQ?download=1)
[X152](./configs/d2/test-d2-X152.yaml)|d2|10.7|**17.7**|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EdLhYc39P8tBkEDVCDOrNV4BgPhz9M4iBq8oPw1iyVSlmg?download=1)


## License

This project is released under the [Apache 2.0 license](LICENSE).

## Contact

This repository is currently maintained by Zhou Yu ([@yuzcccc](https://github.com/yuzcccc)), Tongan Luo ([@Zoroaster97](https://github.com/Zoroaster97)), and Jing Li ([@J1mL3e_](https://github.com/JimLee4530)).

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
