# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.utils.events import get_event_storage
from detectron2.modeling import ROI_HEADS_REGISTRY, ROIHeads
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.backbone.resnet import BottleneckBlock
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.layers import get_norm, BatchNorm2d

from .fast_rcnn import BUACaffeFastRCNNOutputs, BUACaffeFastRCNNOutputLayers, BUADetection2FastRCNNOutputs, BUADetectron2FastRCNNOutputLayers
from .box_regression import BUABox2BoxTransform
from .backbone import BottleneckBlockv2

def make_stage(block_class, num_blocks, first_stride, **kwargs):
    """
    Create a resnet stage by creating many blocks.
    Args:
        block_class (class): a subclass of ResNetBlockBase
        num_blocks (int):
        first_stride (int): the stride of the first block. The other blocks will have stride=1.
            A `stride` argument will be passed to the block constructor.
        kwargs: other arguments passed to the block constructor.

    Returns:
        list[nn.Module]: a list of block module.
    """
    blocks = []
    for i in range(num_blocks):
        if kwargs["dilation"] > 1:
            first_stride = 1
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks

@ROI_HEADS_REGISTRY.register()
class BUACaffeRes5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        # super().__init__(cfg, input_shape)
        super().__init__(cfg)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution     = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type           = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales         = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio        = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.resnet_version   = cfg.MODEL.BUA.RESNET_VERSION
        self.attr_on          = cfg.MODEL.BUA.ATTRIBUTE_ON
        self.extract_on       = cfg.MODEL.BUA.EXTRACT_FEATS
        self.num_attr_classes = cfg.MODEL.BUA.ATTRIBUTE.NUM_CLASSES
        self.extractor_mode   = cfg.MODEL.BUA.EXTRACTOR.MODE

        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.box2box_transform = BUABox2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        self.res5, out_channels = self._build_res5_block(cfg)
        if self.resnet_version == 2:
            self.res5_bn = BatchNorm2d(out_channels, eps=2e-5)
        self.box_predictor = BUACaffeFastRCNNOutputLayers(
            out_channels, self.num_classes, self.cls_agnostic_bbox_reg, attr_on=self.attr_on, num_attr_classes=self.num_attr_classes
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        dilation             = cfg.MODEL.RESNETS.RES5_DILATION
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on
        blocks = make_stage(
            BottleneckBlock if self.resnet_version == 1 else BottleneckBlockv2,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
            dilation=dilation,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        if self.resnet_version == 2:
            out = self.res5[0].conv1(x)
            out = self.res5[0].conv2(out)
            out = self.res5[0].conv3(out)
            if self.res5[0].shortcut is not None:
                shortcut = self.res5[0].shortcut(x)
            else:
                shortcut = x
            out += shortcut
            out = self.res5[1:](out)
            return F.relu_(self.res5_bn(out))
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        image_scales = images.image_scales
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        if self.attr_on:
            pred_class_logits, pred_proposal_deltas, attr_scores = self.box_predictor(feature_pooled, proposals)
        else:
            pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled, proposals)
        if not self.extract_on:
            del feature_pooled

        outputs = BUACaffeFastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            image_scales
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            if self.extract_on:
                num_preds_per_image = [len(p) for p in proposals]
                if self.extractor_mode == 1 or self.extractor_mode == 3:
                    if self.attr_on:
                        return proposal_boxes, outputs.predict_probs(), feature_pooled.split(num_preds_per_image, dim=0), attr_scores.split(num_preds_per_image, dim=0)
                    else:
                        return proposal_boxes, outputs.predict_probs(), feature_pooled.split(num_preds_per_image, dim=0)
                elif self.extractor_mode == 2:
                    return outputs.predict_boxes(), outputs.predict_probs()
                else:
                    raise ValueError('BUA.EXTRATOR.MODE ERROR')
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances, {}

@ROI_HEADS_REGISTRY.register()
class BUADetectron2Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        # super().__init__(cfg, input_shape)
        super().__init__(cfg)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution     = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type           = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales         = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio        = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.resnet_version   = cfg.MODEL.BUA.RESNET_VERSION
        self.attr_on          = cfg.MODEL.BUA.ATTRIBUTE_ON
        self.extract_on       = cfg.MODEL.BUA.EXTRACT_FEATS
        self.num_attr_classes = cfg.MODEL.BUA.ATTRIBUTE.NUM_CLASSES
        self.extractor_mode   = cfg.MODEL.BUA.EXTRACTOR.MODE

        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.box2box_transform = BUABox2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        self.res5, out_channels = self._build_res5_block(cfg)
        if self.resnet_version == 2:
            self.res5_bn = BatchNorm2d(out_channels, eps=2e-5)
        self.box_predictor = BUADetectron2FastRCNNOutputLayers(
            out_channels, self.num_classes, self.cls_agnostic_bbox_reg, \
                attr_on=self.attr_on, num_attr_classes=self.num_attr_classes
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes, gt_attributes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_attributes = gt_attributes[matched_idxs, :]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
            gt_clagt_attributes = -torch.ones((len(matched_idxs),16), dtype=torch.int64).cuda()

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs], gt_attributes[sampled_idxs]

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        dilation             = cfg.MODEL.RESNETS.RES5_DILATION
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock if self.resnet_version == 1 else BottleneckBlockv2,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
            dilation=dilation,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        if self.resnet_version == 2:
            out = self.res5[0].conv1(x)
            out = self.res5[0].conv2(out)
            out = self.res5[0].conv3(out)
            if self.res5[0].shortcut is not None:
                shortcut = self.res5[0].shortcut(x)
            else:
                shortcut = x
            out += shortcut
            out = self.res5[1:](out)
            return F.relu_(self.res5_bn(out))
        return self.res5(x)

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes, gt_attributes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes, targets_per_image.gt_attributes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_attributes = gt_attributes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        # image_scales = images.image_scales
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        if self.attr_on:
            pred_class_logits, pred_proposal_deltas, pred_attribute_logits, gt_attributes = self.box_predictor(feature_pooled, proposals)
        else:
            pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled, proposals)
        if not self.extract_on:
            del feature_pooled
            
        if self.attr_on:
            outputs = BUADetection2FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                self.attr_on,
                pred_attribute_logits=pred_attribute_logits,
                num_attr_classes=self.num_attr_classes,
                gt_attributes=gt_attributes,
            )
        else:
            outputs = BUADetection2FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                self.attr_on,
            )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            if self.extract_on:
                num_preds_per_image = [len(p) for p in proposals]
                if self.extractor_mode == 1 or self.extractor_mode == 3:
                    if self.attr_on:
                        return proposal_boxes, outputs.predict_probs(), feature_pooled.split(num_preds_per_image, dim=0), F.softmax(pred_attribute_logits, dim=-1).split(num_preds_per_image, dim=0)
                    else:
                        return proposal_boxes, outputs.predict_probs(), feature_pooled.split(num_preds_per_image, dim=0)
                elif self.extractor_mode == 2:
                    return outputs.predict_boxes(), outputs.predict_probs()
                else:
                    raise ValueError('BUA.EXTRATOR.MODE ERROR')
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances, {}
