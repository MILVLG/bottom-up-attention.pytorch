import torch
import numpy as np
import cv2
import os

from bua.caffe.modeling.layers.nms import nms
from bua.caffe.modeling.box_regression import BUABoxes

PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
TEST_SCALES = (600,)
TEST_MAX_SIZE = 1000

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def get_image_blob(im, pixel_means):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    pixel_means = np.array([[pixel_means]])
    dataset_dict = {}
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= pixel_means

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    for target_size in TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:
            im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

    dataset_dict["image"] = torch.from_numpy(im).permute(2, 0, 1)
    dataset_dict["im_scale"] = im_scale

    return dataset_dict


def save_roi_features(args, cfg, im_file, im, dataset_dict, boxes, scores, features_pooled, attr_scores=None):
    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH
  
    dets = boxes[0] / dataset_dict['im_scale']
    scores = scores[0]
    feats = features_pooled[0]

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
    image_objects_conf = np.max(scores[keep_boxes].numpy()[:,1:], axis=1)
    image_objects = np.argmax(scores[keep_boxes].numpy()[:,1:], axis=1)
    if not attr_scores is None:
        attr_scores = attr_scores[0]
        image_attrs_conf = np.max(attr_scores[keep_boxes].numpy()[:,1:], axis=1)
        image_attrs = np.argmax(attr_scores[keep_boxes].numpy()[:,1:], axis=1)
        info = {
            'image_id': im_file.split('.')[0],
            'image_h': np.size(im, 0),
            'image_w': np.size(im, 1),
            'num_boxes': len(keep_boxes),
            'objects_id': image_objects,
            'objects_conf': image_objects_conf,
            'attrs_id': image_attrs,
            'attrs_conf': image_attrs_conf,
            }
    else:
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

def save_bbox(args, cfg, im_file, im, dataset_dict, boxes, scores):
    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

    scores = scores[0]
    boxes = boxes[0]
    num_classes = scores.shape[1]
    boxes = BUABoxes(boxes.reshape(-1, 4))
    boxes.clip((dataset_dict['image'].shape[1]/dataset_dict['im_scale'], dataset_dict['image'].shape[2]/dataset_dict['im_scale']))
    boxes = boxes.tensor.view(-1, num_classes*4)  # R x C x 4

    cls_boxes = torch.zeros((boxes.shape[0], 4))
    for idx in range(boxes.shape[0]):
        cls_idx = torch.argmax(scores[idx, 1:]) + 1
        cls_boxes[idx, :] = boxes[idx, cls_idx * 4:(cls_idx + 1) * 4]

    max_conf = torch.zeros((scores.shape[0])).to(scores.device)
    for cls_ind in range(1, num_classes):
            cls_scores = scores[:, cls_ind]
            keep = nms(cls_boxes, cls_scores, 0.3)
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                             cls_scores[keep],
                                             max_conf[keep])
            
    keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]
    image_bboxes = cls_boxes[keep_boxes]

    output_file = os.path.join(args.output_dir, im_file.split('.')[0])
    np.savez_compressed(output_file, bbox=image_bboxes, num_bbox=len(keep_boxes), image_h=np.size(im, 0), image_w=np.size(im, 1))

def save_roi_features_by_bbox(args, cfg, im_file, im, dataset_dict, boxes, scores, features_pooled, attr_scores=None):
    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH
    dets = boxes[0] / dataset_dict['im_scale']
    scores = scores[0]
    feats = features_pooled[0]
    keep_boxes = [i for i in range(scores.shape[0])]

    image_feat = feats[keep_boxes]
    image_bboxes = dets[keep_boxes]
    image_objects_conf = np.max(scores[keep_boxes].numpy()[:,1:], axis=1)
    image_objects = np.argmax(scores[keep_boxes].numpy()[:,1:], axis=1)
    if not attr_scores is None:
        attr_scores = attr_scores[0]
        image_attrs_conf = np.max(attr_scores[keep_boxes].numpy()[:,1:], axis=1)
        image_attrs = np.argmax(attr_scores[keep_boxes].numpy()[:,1:], axis=1)
        info = {
            'image_id': im_file.split('.')[0],
            'image_h': np.size(im, 0),
            'image_w': np.size(im, 1),
            'num_boxes': len(keep_boxes),
            'objects_id': image_objects,
            'objects_conf': image_objects_conf,
            'attrs_id': image_attrs,
            'attrs_conf': image_attrs_conf,
            }
    else:
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
