# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from .load_vg_json import load_vg_json

SPLITS = {
    "visual_genome_train": ("vg/images", "vg/annotations/train.json"),
    "visual_genome_val": ("vg/images", "vg/annotations/val.json"),
}

for key, (image_root, json_file) in SPLITS.items():
    # Assume pre-defined datasets live in `./datasets`.
    json_file = os.path.join("datasets", json_file)
    image_root = os.path.join("datasets", image_root)

    DatasetCatalog.register(
        key,
        lambda key=key, json_file=json_file, image_root=image_root: load_vg_json(
            json_file, image_root, key
        ),
    )

    MetadataCatalog.get(key).set(
        json_file=json_file, image_root=image_root
    )
