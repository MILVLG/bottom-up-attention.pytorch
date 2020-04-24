from .bua_caffe import add_bottom_up_attention_config

def add_config(args, cfg):
    if args.model == "bua_caffe":
        add_bottom_up_attention_config(cfg, True)
    elif args.model == "bua_detectron2":
        add_bottom_up_attention_config(cfg)
    else:
        raise Exception("detection model not supported: {}".format(args.model))
