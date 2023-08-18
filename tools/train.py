import os
import os.path as osp
import time
import argparse
import torch
import torch.nn as nn
import resource
import datetime
from mmcv import Config, mkdir_or_exist
import mmcv
import mlltn
from mlltn.train import train_classifier
from mllt.datasets import build_dataset
from mllt.apis import (set_random_seed, freeze_layer)
from mllt.models import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument(
        'config', help='train config file path')
    parser.add_argument(
        '--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--warmup_epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument(
        '--local_rank', type=int, default=0)
    parser.add_argument(
        '--random_head_alpha', type=float, default=None)
    parser.add_argument(
        '--random_medium_alpha', type=float, default=None)
    parser.add_argument(
        '--random_tail_alpha', type=float, default=None)
    parser.add_argument(
        '--random_head_beta', type=float, default=None)
    parser.add_argument(
        '--random_medium_beta', type=float, default=None)
    parser.add_argument(
        '--random_tail_beta', type=float, default=None)
    parser.add_argument(
        '--balance_head_alpha', type=float, default=None)
    parser.add_argument(
        '--balance_medium_alpha', type=float, default=None)
    parser.add_argument(
        '--balance_tail_alpha', type=float, default=None)
    parser.add_argument(
        '--balance_head_beta', type=float, default=None)
    parser.add_argument(
        '--balance_medium_beta', type=float, default=None)
    parser.add_argument(
        '--balance_tail_beta', type=float, default=None)
    args, _  = parser.parse_known_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # update configs according to CLI args
    mlltn.update_config(cfg, vars(args))

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # save config file to work dir
    mkdir_or_exist(cfg.work_dir)
    os.system('cp {} {}'.format(args.config, cfg.work_dir))

    # init logger before other steps
    mkdir_or_exist(os.path.join(cfg.work_dir, 'logs'))
    now_str = datetime.datetime.now().__str__().replace(' ', '_')
    logfile = os.path.join(cfg.work_dir, 'logs', 'LOG_TRAIN_'+now_str+'.txt')
    logger = mmcv.get_logger('',
                             logfile,
                             cfg.log_level)

    logger.info('Args: {}'.format(args))
    logger.info('Train_cfg: {}'.format(cfg.train_cfg))
    logger.info('Config: {}'.format(cfg))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seedllL to {}'.format(args.seed))
        set_random_seed(args.seed)

    train_dataset = build_dataset(cfg.data.train)

    model = build_classifier(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model = freeze_layer(model)
    # add an attribute for visualization convenience
    model.CLASSES = train_dataset.CLASSES

    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)

    train_classifier(
        model, train_dataset, cfg, logger=logger)

    logger.info(cfg.work_dir)


if __name__ == '__main__':
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))
    main()
