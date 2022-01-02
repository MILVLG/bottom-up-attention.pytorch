from .d2 import add_attribute_config
from .caffe import add_bottom_up_attention_config

def add_config(args, cfg):
    if args.mode == "caffe":
        add_bottom_up_attention_config(cfg, True)
    elif args.mode == "d2":
        add_attribute_config(cfg)
    else:
        raise Exception("detection model not supported: {}".format(args.model))
from . import visual_genome