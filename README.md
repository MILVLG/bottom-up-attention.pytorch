# bottom-up-attention.pytorch

An PyTorch reimplementation of bottom-up-attention models. 

This project not only transfored the bottom-up-attention model to the Detectron 2, but also migrates the original test indicators so that the new model can be compared with the bottom-up-attention Model.

## Installation

**Requirements:**

- Python 3.6
- opencv-python
- Detectron2

**installation:**

```python setup.py build_ext```

## Pretrained models

checkout [MODEL_ZOO.md](MODEL_ZOO.md)

Our Project transfers the weights and models to detectron2. It can extract the features consistent with the bottom-up-attention Model base Caffe.

## Visual Genome Dataset

We process the VG dataset to make it similar to the COCO data format. It can be found in [link]().

## Test & Extract Feature

**Testing:**

```python train_net.py --model bua_detectron2 --config-file configs/test-bua-caffe-r101.yaml --eval-only```

**Extract Feature:**

```python extract_feature.py --config-file configs/extract-bua-caffe-r101.yaml --image_dir image_dir_path --out_dir out_dir_path --resume```

## Reference

Detectron2:

```
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

Bottom-up Attention:

```
@inproceedings{Anderson2017up-down,
  author = {Peter Anderson and Xiaodong He and Chris Buehler and Damien Teney and Mark Johnson and Stephen Gould and Lei Zhang},
  title = {Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering},
  booktitle={CVPR},
  year = {2018}
}
```