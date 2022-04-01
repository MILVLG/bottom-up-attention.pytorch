# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pylint: disable=no-member
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import argparse
import os
import sys
import torch
# import tqdm
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
from utils.extract_utils import get_image_blob, save_bbox, save_roi_features_by_bbox, save_roi_features
from utils.progress_bar import ProgressBar
from bua import add_config
from bua.caffe.modeling.box_regression import BUABoxes

import ray
from ray.actor import ActorHandle

"""
add ray to generate_npz
"""
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

def set_min_max_boxes(min_max_boxes):
    if min_max_boxes == 'min_max_default':
        return []
    try:
        min_boxes = int(min_max_boxes.split(',')[0])
        max_boxes = int(min_max_boxes.split(',')[1])
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
    cfg.merge_from_list(set_min_max_boxes(args.min_max_boxes))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

@ray.remote
def generate_npz(extract_mode, pba: ActorHandle, *args):
    if extract_mode == 1:
        save_roi_features(*args)
    elif extract_mode == 2:
        save_bbox(*args)
    elif extract_mode == 3:
        save_roi_features_by_bbox(*args)
    else:
        print('Invalid Extract Mode! ')
    pba.update.remote(1)

@ray.remote(num_gpus=1)
def extract_feat_faster(split_idx, img_list, cfg, args, actor: ActorHandle):
    num_images = len(img_list)
    print('Number of images on split{}: {}.'.format(split_idx, num_images))

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()

    generate_npz_list = []
    for im_file in (img_list):
        if os.path.exists(os.path.join(args.output_dir, im_file.split('.')[0]+'.npz')):
            actor.update.remote(1)
            continue
        im = cv2.imread(os.path.join(args.image_dir, im_file))
        if im is None:
            print(os.path.join(args.image_dir, im_file), "is illegal!")
            actor.update.remote(1)
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
            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
            if not attr_scores is None:
                attr_scores = [attr_score.cpu() for attr_score in attr_scores]
            generate_npz_list.append(generate_npz.remote(1, actor, 
                args, cfg, im_file, im, dataset_dict, 
                boxes, scores, features_pooled, attr_scores))
        # extract bbox only
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 2:
            with torch.set_grad_enabled(False):
                boxes, scores = model([dataset_dict])
            boxes = [box.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            generate_npz_list.append(generate_npz.remote(2, actor, 
                args, cfg, im_file, im, dataset_dict, 
                boxes, scores))
        # extract roi features by bbox
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 3:
            if not os.path.exists(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz')):
                actor.update.remote(1)
                continue
            bbox = torch.from_numpy(np.load(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz'))['bbox']) * dataset_dict['im_scale']
            proposals = Instances(dataset_dict['image'].shape[-2:])
            proposals.proposal_boxes = BUABoxes(bbox)
            dataset_dict['proposals'] = proposals

            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model([dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])
            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
            if not attr_scores is None:
                attr_scores = [attr_score.data.cpu() for attr_score in attr_scores]
            generate_npz_list.append(generate_npz.remote(3, actor, 
                args, cfg, im_file, im, dataset_dict, 
                boxes, scores, features_pooled, attr_scores))

    ray.get(generate_npz_list)


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
    extract_feat_faster_start(args,cfg)

def extract_feat_faster_start(args,cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    num_gpus = len(args.gpu_id.split(','))

    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

    # Extract features.
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)
    print('Number of images: {}.'.format(num_images))

    if args.num_cpus != 0:
        ray.init(num_cpus=args.num_cpus)
    else:
        ray.init()
    img_lists = [imglist[i::num_gpus] for i in range(num_gpus)]

    pb = ProgressBar(len(imglist))
    actor = pb.actor

    print('Number of GPUs: {}.'.format(num_gpus))
    extract_feat_list = []
    for i in range(num_gpus):
        extract_feat_list.append(extract_feat_faster.remote(i, img_lists[i], cfg, args, actor))
    
    pb.print_until_done()
    ray.get(extract_feat_list)
    ray.get(actor.get_counter.remote())

if __name__ == "__main__":
    main()