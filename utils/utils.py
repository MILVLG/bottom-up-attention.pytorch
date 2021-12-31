import os
import errno
import numpy as np
import torch

from detectron2.structures import Instances
from bua.caffe.modeling.layers.nms import nms

def save_features(output_file, features, boxes=None):
    if boxes is None:
        res = features
        np.save(output_file, res)
    else:
        np.savez(output_file, x=features, bbox=boxes)

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def extractor_postprocess(boxes, scores, features_pooled, input_per_image, extractor):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    MIN_BOXES = extractor.MIN_BOXES
    MAX_BOXES = extractor.MAX_BOXES
    CONF_THRESH = extractor.CONF_THRESH

    cur_device = scores.device

    dets = boxes / input_per_image["im_scale"]

    max_conf = torch.zeros((scores.shape[0])).to(cur_device)

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
    image_feat = features_pooled[keep_boxes]
    image_bboxes = dets[keep_boxes]

    return image_feat, image_bboxes