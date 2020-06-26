# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import argparse
import os
import sys
import torch
import tqdm
import cv2
import numpy as np
sys.path.append('detectron2')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.structures import Instances

from utils.utils import mkdir, save_features
from utils.extract_utils import get_image_blob, save_bbox, save_roi_features_by_gt_bbox, save_roi_features
from models import add_config
from models.bua.box_regression import BUABoxes

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
    parser.add_argument(
        "--config-file",
        default="configs/bua-caffe/extract-bua-caffe-r101.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--mode", default="caffe", type=str, help="bua_caffe, ...")

    parser.add_argument('--out-dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--image-dir', dest='image_dir',
                        help='directory with images',
                        default="image")
    parser.add_argument('--gt-bbox-dir', dest='gt_bbox_dir',
                        help='directory with gt-bbox',
                        default="bbox")
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

    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    # Extract features.
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)
    print('Number of images: {}.'.format(num_images))
    model.eval()

    for im_file in tqdm.tqdm(imglist):
        if os.path.exists(os.path.join(args.output_dir, im_file.split('.')[0]+'.npz')):
            continue
        im = cv2.imread(os.path.join(args.image_dir, im_file))
        if im is None:
            print(os.path.join(args.image_dir, im_file), "is illegal!")
            continue
        dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
        # extract roi features
        if cfg.MODEL.BUA.EXTRACTOR.MODE == 1:
            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model([dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])
            save_roi_features(args, cfg, im_file, im, dataset_dict, boxes, scores, features_pooled, attr_scores)
            # extract bbox only
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 2:
            with torch.set_grad_enabled(False):
                boxes, scores = model([dataset_dict])
            save_bbox(args, cfg, im_file, im, dataset_dict, boxes, scores)
            # extract roi features by gt bbox
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 3:
            if not os.path.exists(os.path.join(args.gt_bbox_dir, im_file.split('.')[0]+'.npz')):
                continue
            bbox = torch.from_numpy(np.load(os.path.join(args.gt_bbox_dir, im_file.split('.')[0]+'.npz'))['bbox']) * dataset_dict['im_scale']
            proposals = Instances(dataset_dict['image'].shape[-2:])
            proposals.proposal_boxes = BUABoxes(bbox)
            dataset_dict['proposals'] = proposals

            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model([dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])

            save_roi_features_by_gt_bbox(args, cfg, im_file, im, dataset_dict, boxes, scores, features_pooled, attr_scores)
         

if __name__ == "__main__":
    main()
