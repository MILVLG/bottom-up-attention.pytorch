from .dataloader.build_loader import (
    build_detection_train_loader_with_attributes,
    build_detection_test_loader_with_attributes,
)
from .modeling.roi_heads import AttributeRes5ROIHeads
from .. import visual_genome
from .config import add_attribute_config