from mmcv.utils.config import ConfigDict


def update_item(cfg, k, v):
    if k in cfg:
        if v is not None:
            cfg[k] = v
        return
    for sub_cfg in cfg:
        if isinstance(cfg[sub_cfg], ConfigDict):
            update_item(cfg[sub_cfg], k, v)


def update_config(cfg, params):
    for k, v in params.items():
        update_item(cfg, k, v)
