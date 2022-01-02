# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import sys
import time
sys.path.append('detectron2')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from bua import add_config
from bua.d2 import build_detection_test_loader_with_attributes, build_detection_train_loader_with_attributes
from bua.caffe.dataloader import DatasetMapper
from opts import parse_opt
from evaluation import VGEvaluator


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rpn_box_lw = cfg.MODEL.RPN.BBOX_LOSS_WEIGHT
        self.rcnn_box_lw = cfg.MODEL.ROI_BOX_HEAD.BBOX_LOSS_WEIGHT

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return VGEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.MODE == "caffe":
            return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))
        elif cfg.MODE == "d2":
            return build_detection_test_loader_with_attributes(cfg, dataset_name)
        else:
            raise Exception("detectron mode note supported: {}".format(args.model))
        


    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODE == "caffe":
            return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
        elif cfg.MODE == "d2":
            return build_detection_train_loader_with_attributes(cfg)
        else:
            raise Exception("detectron mode note supported: {}".format(args.model))

    def run_step(self):
        """
        !!Hack!! for the run_step method in SimpleTrainer to adjust the loss
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        loss_dict = self.model(data)
        # RPN box loss:
        loss_dict["loss_rpn_loc"] *= self.rpn_box_lw
        # R-CNN box loss:
        loss_dict["loss_box_reg"] *= self.rcnn_box_lw
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODE = args.mode
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = parse_opt().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
