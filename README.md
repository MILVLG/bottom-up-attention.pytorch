# bottom-up-attention.pytorch

An PyTorch reimplementation of [bottom-up-attention models](https://github.com/peteanderson80/bottom-up-attention). 

This project not only transfored the bottom-up-attention models to the [Detectron2](https://github.com/facebookresearch/detectron2), but also migrates the original test indicators so that the new model can be compared with the bottom-up-attention models.

## Installation

**Requirements:**

- Python 3.6
- opencv-python
- Detectron2

**installation:**

1. You need to install detectron2 according to the [installation instructions](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

2. ```python setup.py build_ext```

## Pretrained models

Our Project transfers the weights and models to detectron2. It can extract the features consistent with the bottom-up-attention Model base Caffe.

| |bacbone|objects mAP@0.5|objects weighted mAP@0.5|
-|-|-|-
|[Faster R-CNN](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EaORTX7eenZOgGDjKe03e6UB31ty7Q2bkAN-LEKrqjSa6A?e=6iQGAj)|ResNet-101|10.2%|15.1%|

Note that mAP is relatively low because many classes overlap (e.g. person / man / guy), some classes can't be precisely located (e.g. street, field) and separate classes exist for singular and plural objects (e.g. person / people). We focus on performance in downstream tasks (e.g. image captioning, VQA) rather than detection performance.

## Visual Genome Dataset

We process the VG dataset to make it similar to the COCO data format. It can be found in [link](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EWpiE_5PvBdKiKfCi0pBx_EB5ONo8D8XABUz7tWcnltCrw?e=xIeW23).

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

## Test & Extract Feature

**Testing:**

```python train_net.py --model bua_caffe --config-file configs/bua-caffe/test-bua-caffe-r101.yaml --eval-only --resume```

To add:

1. ```--model```,e.g.```--model bua_caffe``` to assign a model for testing.

2. ```--config-file```,e.g.```--config-file configs/bua-caffe/test-bua-caffe-r101.yaml``` to import configuration file.

3. ```--eval-only``` to run eval only.

4. ```--resume``` to start training with saved checkpoint parameters. 

**Extract Feature:**

```python extract_feature.py --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml --image_dir image_dir_path --out_dir out_dir_path --resume```

To add:

1. ```--config-file```,e.g.```--config-file configs/bua-caffe/test-bua-caffe-r101.yaml``` to import configuration file.

2. ```--image_dir``` to assign image dir.

3. ```--out_dir``` to assign output dir.

4. ```--resume``` to start training with saved checkpoint parameters. 