import os, io
import numpy as np

import copy
import torch
import logging
import pickle as cPickle
import itertools
import contextlib
from pycocotools.coco import COCO
from collections import OrderedDict
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

from .vg_eval import vg_eval

class VGEvaluator(DatasetEvaluator):
    """
        Evaluate object proposal, instance detection
        outputs using VG's metrics and APIs.
    """
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device("cpu")
        self._output_dir = output_dir

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = os.path.join(output_dir, f"{dataset_name}_vg_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._classes = ['__background__']
        self._class_to_ind = {}
        self._class_to_ind[self._classes[0]] = 0
        with open(os.path.join('evaluation/objects_vocab.txt')) as f:
            count = 1
            for object in f.readlines():
                names = [n.lower().strip() for n in object.split(',')]
                self._classes.append(names[0])
                for n in names:
                    self._class_to_ind[n] = count
                count += 1

        # Load attributes
        self._attributes = ['__no_attribute__']
        self._attribute_to_ind = {}
        self._attribute_to_ind[self._attributes[0]] = 0
        with open(os.path.join('evaluation/attributes_vocab.txt')) as f:
            count = 1
            for att in f.readlines():
                names = [n.lower().strip() for n in att.split(',')]
                self._attributes.append(names[0])
                for n in names:
                    self._attribute_to_ind[n] = count
                count += 1

        self.roidb, self.image_index = self.gt_roidb(self._coco_api)

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def gt_roidb(self, dataset):
        roidb = []
        image_index = dataset.imgToAnns.keys()
        for img_index in dataset.imgToAnns:
            tmp_dict = {}
            num_objs = len(dataset.imgToAnns[img_index])
            bboxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_attributes = np.zeros((num_objs, 16), dtype=np.int32)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            for ind, item in enumerate(dataset.imgToAnns[img_index]):
                bboxes[ind, :] = item['bbox']
                gt_classes[ind] = item['category_id'] + 1 # NOTE
                for j, attr in enumerate(item['attribute_ids']):
                    gt_attributes[ind, j] = attr
            bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
            tmp_dict['boxes'] = bboxes
            tmp_dict['gt_attributes'] = gt_attributes
            tmp_dict['gt_classes'] = gt_classes
            roidb.append(tmp_dict)
        return roidb, image_index

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["boxes"] = instances.pred_boxes.tensor.numpy()
                prediction["labels"] = instances.pred_classes.numpy()
                prediction["scores"] = instances.scores.numpy()
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        # self._predictions = torch.load(os.path.join(self._output_dir, "instances_predictions.pth"))

        if len(self._predictions) == 0:
            self._logger.warning("[VGEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        self._eval_vg()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_vg(self):
        self.write_voc_results_file(self._predictions, output_dir=self._output_dir)
        self.do_python_eval(self._output_dir)

    def write_voc_results_file(self, predictions, output_dir):

        # preds = []
        # for item in predictions:
        #     pred = {}
        #     pred['image_id'] = item['image_id']
        #     scores = item["scores"]
        #     labels = item["labels"]
        #     bbox = item["boxes"]
        #     for ind, instance in enumerate(item['instances']):
        #         scores[ind] = instance['score']
        #         labels[ind] = instance['category_id']
        #         bbox[ind, :] = instance['bbox'][:]
        #     pred['scores'] = scores
        #     pred['lables'] = labels
        #     pred['bbox'] = bbox
        #     preds.append(pred)

        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('Writing "{}" vg result file'.format(cls))
            filename = self.get_vg_results_file_template(output_dir).format(cls)
            with open(filename, 'wt') as f:
                for pred_ind, item in enumerate(predictions):
                    scores = item["scores"]
                    labels = item["labels"]+1
                    bbox = item["boxes"]
                    if cls_ind not in labels:
                        continue
                    dets = bbox[labels==cls_ind]
                    scores = scores[labels==cls_ind]
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(item["image_id"]), scores[k],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def get_vg_results_file_template(self, output_dir, pickle=True, eval_attributes = False):
        filename = 'detections_vg'+'_{:s}.txt'
        path = os.path.join(output_dir, filename)
        return path

    def do_python_eval(self, output_dir, pickle=True, eval_attributes = False):
        # We re-use parts of the pascal voc python code for visual genome
        aps = []
        nposs = []
        thresh = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # Load ground truth
        if eval_attributes:
            classes = self._attributes
        else:
            classes = self._classes
        for i, cls in enumerate(classes):
            if cls == '__background__' or cls == '__no_attribute__':
                continue
            filename = self.get_vg_results_file_template(output_dir).format(cls)
            rec, prec, ap, scores, npos = vg_eval(
                filename, self.roidb, self.image_index, i, ovthresh=0.5,
                use_07_metric=use_07_metric, eval_attributes=eval_attributes)

            # Determine per class detection thresholds that maximise f score
            if npos > 1 and not (type(prec) == int and type(rec) == int and prec+rec ==0):
                f = np.nan_to_num((prec * rec) / (prec + rec))
                thresh += [scores[np.argmax(f)]]
            else:
                thresh += [0]
            aps += [ap]
            nposs += [float(npos)]
            print('AP for {} = {:.4f} (npos={:,})'.format(cls, ap, npos))
            if pickle:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap,
                                  'scores': scores, 'npos': npos}, f)

        # Set thresh to mean for classes with poor results
        thresh = np.array(thresh)
        avg_thresh = np.mean(thresh[thresh != 0])
        thresh[thresh == 0] = avg_thresh
        if eval_attributes:
            filename = 'attribute_thresholds_vg.txt'
        else:
            filename = 'object_thresholds_vg.txt'
        path = os.path.join(output_dir, filename)
        with open(path, 'wt') as f:
            for i, cls in enumerate(classes[1:]):
                f.write('{:s} {:.3f}\n'.format(cls, thresh[i]))

        weights = np.array(nposs)
        weights /= weights.sum()
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('Weighted Mean AP = {:.4f}'.format(np.average(aps, weights=weights)))
        print('Mean Detection Threshold = {:.3f}'.format(avg_thresh))
        # print('~~~~~~~~')
        # print('Results:')
        # for ap, npos in zip(aps, nposs):
        #     print('{:.3f}\t{:.3f}'.format(ap, npos))
        # print('{:.3f}'.format(np.mean(aps)))
        # print('~~~~~~~~')
        # print('')
        # print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** PASCAL VOC Python eval code.')
        print('--------------------------------------------------------------')
