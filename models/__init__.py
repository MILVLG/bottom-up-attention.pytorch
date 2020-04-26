from .bua import add_bottom_up_attention_config

def add_config(args, cfg):
    if args.mode == "caffe":
        add_bottom_up_attention_config(cfg, True)
    elif args.mode == "detectron2":
        add_bottom_up_attention_config(cfg)
    else:
        raise Exception("detection model not supported: {}".format(args.model))
