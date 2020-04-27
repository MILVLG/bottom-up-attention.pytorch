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
import base64
sys.path.append('detectron2')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from utils.utils import mkdir, save_features
from utils.extract_utils import get_image_blob
from models import add_config
from models.bua.layers.nms import nms

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

    MIN_BOXES = 10
    MAX_BOXES = 100
    CONF_THRESH = 0.2

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
        im = cv2.imread(os.path.join(args.image_dir, im_file))
        dataset_dict = get_image_blob(im)

        with torch.set_grad_enabled(False):
            boxes, scores, features_pooled = model([dataset_dict])

        dets = boxes[0].tensor.cpu() / dataset_dict['im_scale']
        scores = scores[0].cpu()
        feats = features_pooled[0].cpu()

        max_conf = torch.zeros((scores.shape[0])).to(scores.device)
        for cls_ind in range(1, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.3)
                max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                             cls_scores[keep],
                                             max_conf[keep])
            
        keep_boxes = torch.nonzero(max_conf >= CONF_THRESH).flatten()
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]
        image_feat = feats[keep_boxes]
        image_bboxes = dets[keep_boxes]
        image_objects_conf = np.max(scores[keep_boxes].numpy(), axis=1)
        image_objects = np.argmax(scores[keep_boxes].numpy(), axis=1)
        info = {
        'image_id': im_file.split('.')[0],
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'objects_id': image_objects,
        'objects_conf': image_objects_conf
        }  

        output_file = os.path.join(args.output_dir, im_file.split('.')[0])
        np.savez_compressed(output_file, x=image_feat, bbox=image_bboxes, num_bbox=len(keep_boxes), image_h=np.size(im, 0), image_w=np.size(im, 1), info=info) 

if __name__ == "__main__":
    main()
