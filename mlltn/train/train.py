from __future__ import division

import os
import re
from collections import OrderedDict

import torch
from mmcv.runner import EpochBasedRunner, DistSamplerSeedHook, obj_from_dict
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner.checkpoint import load_checkpoint
from mmcv import mkdir_or_exist

from mllt.datasets import build_dataloader
from mllt.apis import build_optimizer, get_root_logger, load_certain_checkpoint, parse_losses
from .dataloaders import DualLoader
from mmcv.runner import Hook
import logging


class SamplerSeedHook(Hook):

    def before_epoch(self, runner):

        runner.data_loader.sampler.set_epoch(runner.epoch)
        if hasattr(runner.model, 'module') and hasattr(runner.model.module, 'set_epoch'):
            runner.model.module.set_epoch(runner.epoch)

    def after_epoch(self, runner):

        if hasattr(runner.data_loader.sampler, 'get_weights'):
            logging.info(runner.data_loader.sampler.get_weights())
        if hasattr(runner.data_loader.sampler, 'reset_weights'):
            runner.data_loader.sampler.reset_weights(runner.epoch)


def prefix_loss(losses, prefix):

    new_losses = {}

    for k, v in losses.items():
        new_losses['{}-branch/{}'.format(prefix, k)] = v

    return new_losses


def batch_processor(model, data, train_mode):

    random_data = data[0]
    balanced_data = data[1]

    losses_random = model(branch='random',
                          data2=random_data[1],
                          **random_data[0])
    losses_balanced = model(branch='balance',
                            data2=balanced_data[1],
                            **balanced_data[0])

    losses_random = prefix_loss(losses_random, 'random')
    losses_balanced = prefix_loss(losses_balanced, 'balance')

    loss, log_vars = parse_losses({**losses_random, **losses_balanced})

    outputs = dict(
        loss=loss, log_vars=log_vars,
        num_samples=len(random_data[0]['img'].data)+len(balanced_data[0]['img'].data))

    return outputs


def train_classifier(model,
                         dataset,
                         cfg,
                         logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    _non_dist_train(model, dataset, cfg, logger=logger)


def _non_dist_train(model, dataset, cfg, logger=None):
    # prepare data loaders
    sampler_cfg = cfg.data.get('sampler_cfg', None)
    sampler = cfg.data.get('sampler', 'Group')
    sampler_cfg1 = cfg.data.get('sampler_cfg1', None)
    sampler1 = cfg.data.get('sampler1', 'Group')
    factor = cfg.data.get('data_factor', 8)
    bz1 = cfg.data.imgs_per_gpu if sampler == 'None' else cfg.data.imgs_per_gpu*factor
    bz2 = cfg.data.imgs_per_gpu if sampler1 == 'None' else cfg.data.imgs_per_gpu*factor

    loader1 = build_dataloader(
        dataset,
        bz1,
        cfg.data.workers_per_gpu,
        cfg.gpus,
        dist=False,
        sampler=sampler,
        sampler_cfg=sampler_cfg)
    loader2 = build_dataloader(
        dataset,
        bz2,
        cfg.data.workers_per_gpu,
        cfg.gpus,
        dist=False,
        sampler=sampler1,
        sampler_cfg=sampler_cfg1)
    data_loaders = [
        DualLoader(loader1, loader2)
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    logging.info([k for k,v in model.named_parameters() if v.requires_grad])

    runner = EpochBasedRunner(model, batch_processor, optimizer,
                              cfg.work_dir, logger)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(SamplerSeedHook())

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        load_modules = cfg.get('load_modules', [])
        load_certain_checkpoint(runner.model, runner.logger, cfg.load_from, load_modules)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
