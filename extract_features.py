# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pylint: disable=no-member
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import argparse
from ast import arg
import os
import sys
import torch
# import tqdm
import cv2
import numpy as np

from utils.extract_d2features import extract_feat_d2_start
sys.path.append('detectron2')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.structures import Instances

from utils.utils import mkdir, save_features
from utils.extract_utils import get_image_blob, save_bbox, save_roi_features_by_bbox, save_roi_features
from utils.progress_bar import ProgressBar
from bua import add_config
from bua.caffe.modeling.box_regression import BUABoxes
from torch.nn import functional as F
from detectron2.modeling import postprocessing
from utils.extract_features_singlegpu import extract_feat_singlegpu_start
from utils.extract_features_multigpu import extract_feat_multigpu_start
from utils.extract_features_faster import extract_feat_faster_start

def switch_extract_mode(mode):
    if mode == 'roi_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 1]
    elif mode == 'bboxes':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 2]
    elif mode == 'bbox_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 3, 'MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
    else:
        print('Wrong extract mode! ')
        exit()
    return switch_cmd
  # ROI_HEADS:  # Add to get 100 box or Delete it to get ~50 boxes
  #   SCORE_THRESH_TEST: 0.0
  #   NMS_THRESH_TEST: 0.3   
def set_min_max_boxes(min_max_boxes, mode):
    if min_max_boxes == 'min_max_default':
        return []
    try:
        min_boxes = int(min_max_boxes.split(',')[0])
        max_boxes = int(min_max_boxes.split(',')[1])
        if mode == "caffe":
            pass
        elif mode == "d2":
            if min_boxes == 100 & max_boxes == 100:
                cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes, 
                        'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes,
                        'MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.0,
                        'MODEL.ROI_HEADS.NMS_THRESH_TEST', 0.3 ]
                return cmd
        else:
            raise Exception("detection mode not supported: {}".format(mode))
    except:
        print('Illegal min-max boxes setting, using config default. ')
        return []
    cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes, 
            'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes]
    return cmd

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(['MODEL.BUA.EXTRACT_FEATS',True])
    cfg.merge_from_list(switch_extract_mode(args.extract_mode))
    cfg.merge_from_list(set_min_max_boxes(args.min_max_boxes, args.mode))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
    parser.add_argument(
        "--config-file",
        default="configs/caffe/test-caffe-r101.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument('--num-cpus', default=1, type=int, 
                        help='number of cpus to use for ray, 0 means no limit')

    parser.add_argument('--gpus', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)

    parser.add_argument("--mode", default="caffe", type=str, help="'caffe' and 'd2' indicates \
                        'use caffe model' and 'use detectron2 model'respectively")

    parser.add_argument('--extract-mode', default='roi_feats', type=str,
                        help="'roi_feats', 'bboxes' and 'bbox_feats' indicates \
                        'extract roi features directly', 'extract bboxes only' and \
                        'extract roi features with pre-computed bboxes' respectively")

    parser.add_argument('--min-max-boxes', default='min_max_default', type=str, 
                        help='the number of min-max boxes of extractor')

    parser.add_argument('--out-dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--image-dir', dest='image_dir',
                        help='directory with images',
                        default="image")
    parser.add_argument('--bbox-dir', dest='bbox_dir',
                        help='directory with bbox',
                        default="bbox")
    parser.add_argument("--fastmode", action="store_true", help="whether to use multi cpus to extract faster.",)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg = setup(args)
    num_gpus = len(args.gpu_id.split(','))
    print(args.mode)
    if args.mode == "caffe":
        if args.fastmode: # faster.py
            print("faster")
            extract_feat_faster_start(args,cfg)
        else:  # multi or single
            if num_gpus == 1: # without ray
                print("single")
                extract_feat_singlegpu_start(args,cfg)
            else: # use ray to accelerate
                print("multi")
                extract_feat_multigpu_start(args,cfg)
    elif args.mode == "d2":
        print("d2 mode use ray")
        extract_feat_d2_start(args,cfg)
    else:
        raise Exception("detection model not supported: {}".format(args.model))

if __name__ == "__main__":
    main()
