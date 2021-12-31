# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_bottom_up_attention_config(cfg, caffe=False):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.BUA = CN()
    _C.MODEL.BUA.CAFFE = caffe
    _C.MODEL.BUA.RESNET_VERSION = 1
    _C.MODEL.BUA.ATTRIBUTE_ON = False
    _C.MODEL.BUA.EXTRACT_FEATS = False

    _C.MODEL.BUA.RPN = CN()
    # out_channels of conv for bottom-up-attentions RPN.
    _C.MODEL.BUA.RPN.CONV_OUT_CHANNELS = 512

    _C.MODEL.BUA.EXTRACTOR = CN()

    # EXTRACTOR.MODE {1: extract roi features, 2: extract bbox only ,3: extract roi features by gt_bbox}
    _C.MODEL.BUA.EXTRACTOR.MODE = 1

    # config of postprocessing in extractor
    _C.MODEL.BUA.EXTRACTOR.MIN_BOXES = 10
    _C.MODEL.BUA.EXTRACTOR.MAX_BOXES = 100
    _C.MODEL.BUA.EXTRACTOR.CONF_THRESH = 0.2
    _C.MODEL.BUA.EXTRACTOR.OUTPUT_DIR = ".output/"

    _C.MODEL.BUA.ATTRIBUTE = CN()
    _C.MODEL.BUA.ATTRIBUTE.NUM_CLASSES = 401
