import torch.nn as nn
import mmcv

from .base import BaseClassifier
from .. import builder
from ..registry import CLASSIFIERS
from mmcv.parallel import DataContainer as DC
import torch
import numpy as np
import copy


@CLASSIFIERS.register_module
class SimpleClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 head,
                 head1,
                 neck=None,
                 neck1=None,
                 lock_back=False,
                 lock_neck=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 savefeat=False):
        super(SimpleClassifier, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.backbone1 = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
            self.neck1 = builder.build_neck(neck1)
        self.head = builder.build_head(head)
        self.head1 = builder.build_head(head1)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.lock_back = lock_back
        self.lock_neck = lock_neck
        self.savefeat = savefeat
        if self.savefeat and not self.with_neck:
            assert neck is not None, 'We must have a neck'
            assert train_cfg is None, 'this is only at testing stage'
        if self.lock_back:
            print('\033[1;35m >>> backbone locked !\033[0;0m')
        if self.lock_neck:
            print('\033[1;35m >>> neck locked !\033[0;0m')
        self.init_weights(pretrained=pretrained)
        self.epoch = 0

        if self.train_cfg is not None:
            self.init_splits()
            self.init_thresholds(self.head.num_classes)

    def init_splits(self):

        x = mmcv.load(self.train_cfg.class_split)
        self.head_list = x['head']
        self.medium_list = x['middle']
        self.tail_list = x['tail']

    def init_thresholds(self, num_classes):

        self.random_alphas = []
        self.random_betas = []
        self.balanced_alphas = []
        self.balanced_betas = []

        for i in range(num_classes):

            if i in self.head_list:
                self.random_alphas.append(self.train_cfg.random_head_alpha)
                self.random_betas.append(self.train_cfg.random_head_beta)
                self.balanced_alphas.append(self.train_cfg.balance_head_alpha)
                self.balanced_betas.append(self.train_cfg.balance_head_beta)
            elif i in self.medium_list:
                self.random_alphas.append(self.train_cfg.random_medium_alpha)
                self.random_betas.append(self.train_cfg.random_medium_beta)
                self.balanced_alphas.append(self.train_cfg.balance_medium_alpha)
                self.balanced_betas.append(self.train_cfg.balance_medium_beta)
            elif i in self.tail_list:
                self.random_alphas.append(self.train_cfg.random_tail_alpha)
                self.random_betas.append(self.train_cfg.random_tail_beta)
                self.balanced_alphas.append(self.train_cfg.balance_tail_alpha)
                self.balanced_betas.append(self.train_cfg.balance_tail_beta)
            else:
                raise Exception('wrong class split!')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def init_weights(self, pretrained=None):
        super(SimpleClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.backbone1.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
                for m in self.neck1:
                    m.init_weights()
            else:
                self.neck.init_weights()
                self.neck1.init_weights()
        self.head.init_weights()
        self.head1.init_weights()

    def extract_feat(self, img, branch=None, need_feat=False, use_bn2=False):
        if self.lock_back:
            with torch.no_grad():
                if branch is None or branch == 'random':
                    x = self.backbone(img)
                elif branch == 'balance':
                    x = self.backbone1(img)
        else:
            if branch is None or branch == 'random':
                x = self.backbone(img)
            elif branch == 'balance':
                x = self.backbone1(img)
        backbone_feat = x[-1]

        if self.with_neck:
            if self.lock_neck:
                with torch.no_grad():
                    x = self.neck(x, use_bn2=use_bn2)
            else:
                if branch is None or branch == 'random':
                    if need_feat:
                        x = self.neck(x, use_bn2=use_bn2)
                        neck_feat = x
                    else:
                        x = self.neck(x, use_bn2=use_bn2)
                elif branch == 'balance':
                    if need_feat:
                        x = self.neck1(x, use_bn2=use_bn2)
                        neck_feat = x
                    else:
                        x = self.neck1(x, use_bn2=use_bn2)
        if need_feat:
            return x, backbone_feat, neck_feat
        return x

    def extract_feat_stitchup(self, img1, img2=None, branch=None, need_feat=False):
        if self.lock_back:
            with torch.no_grad():
                if branch is None or branch in ['random']:
                    x1 = self.backbone(img1)
                    x2 = self.backbone(img2)
                elif branch in ['balance']:
                    x1 = self.backbone1(img1)
                    x2 = self.backbone1(img2)
        else:
            if branch is None or branch in ['random']:
                x1 = self.backbone(img1)
                x2 = self.backbone(img2)
            elif branch in ['balance']:
                x1 = self.backbone1(img1)
                x2 = self.backbone1(img2)

        backbone_feat_img1 = x1
        backbone_feat_img2 = x2

        if self.with_neck:
            if self.lock_neck:
                with torch.no_grad():
                    x = self.neck(x1)
            else:
                if branch is None or branch in ['random']:
                    x, neck_feat1, neck_feat2 = self.neck(x1, x2=x2, need_feat=True)
                elif branch in ['balance']:
                    x, neck_feat1, neck_feat2 = self.neck1(x1, x2=x2, need_feat=True)
        if need_feat:
            return x, neck_feat1, neck_feat2
        return x

    def adjust_labels(self, noisy_labels, prob, branch=None, cls_id=None):

        num_classes = noisy_labels.shape[1]

        assert branch in ['random', 'balance']
        alphas = self.random_alphas if branch == 'random' else self.balanced_alphas
        betas = self.random_betas if branch == 'random' else self.balanced_betas

        new_labels = noisy_labels.clone()

        for i in range(num_classes):

            new_labels[:, i] = torch.where((prob[:, i] > alphas[i]), torch.ones_like(new_labels[:, i]),
                                           new_labels[:, i])
            new_labels[:, i] = torch.where((prob[:, i] < betas[i]), torch.zeros_like(new_labels[:, i]),
                                           new_labels[:, i])

        if self.epoch >= self.train_cfg.warmup_epoch:
            return new_labels, None
        else:
            return noisy_labels, None

    def union_label(self, lb_a, lb_b):

        return torch.where((lb_a > 0)|(lb_b > 0),
                           torch.ones_like(lb_a),
                           torch.zeros_like(lb_a))

    def find_common_label(self, lb_a, lb_b):

        return torch.where((lb_a > 0)&(lb_b > 0),
                           torch.ones_like(lb_a),
                           torch.zeros_like(lb_a))

    def forward_train(self,
                      img,
                      img_metas,
                      noisy_labels,
                      clean_labels=None,
                      branch=None,
                      data2=None):

        x, x1, x2 = self.extract_feat_stitchup(img, data2['img'], branch=branch, need_feat=True)

        if branch is None or branch in ['random']:
            outs = self.head(x)
            outs1 = self.head(x1)
            outs2 = self.head(x2)

            with torch.no_grad():
                self.neck1.eval()
                self.neck.eval()
                y = self.extract_feat(img, 'balance')
                outs_tmp = self.head1(y)
                prob1 = torch.sigmoid(outs_tmp)
                noisy_labels, update_info = self.adjust_labels(noisy_labels, prob1, branch='balance')
                y = self.extract_feat(data2['img'], 'balance')
                outs_tmp = self.head1(y)
                prob1 = torch.sigmoid(outs_tmp)
                data2['noisy_labels'], update_info = \
                    self.adjust_labels(data2['noisy_labels'], prob1, branch='balance')
                self.neck.train()
                self.neck1.train()

        elif branch in ['balance']:
            outs = self.head1(x)
            outs1 = self.head1(x1)
            outs2 = self.head1(x2)

            with torch.no_grad():
                self.neck.eval()
                self.neck1.eval()
                y = self.extract_feat(img, 'random')
                outs_tmp = self.head(y)
                prob1 = torch.sigmoid(outs_tmp)
                noisy_labels, update_info = self.adjust_labels(noisy_labels, prob1, branch='random')
                y = self.extract_feat(data2['img'], 'random')
                outs_tmp = self.head(y)
                prob1 = torch.sigmoid(outs_tmp)
                data2['noisy_labels'], update_info = \
                    self.adjust_labels(data2['noisy_labels'], prob1, branch='random')
                self.neck1.train()
                self.neck.train()

        union_labels = self.union_label(noisy_labels, data2['noisy_labels'])
        common_lbs = self.find_common_label(noisy_labels, data2['noisy_labels'])

        loss_inputs = (outs, union_labels)
        loss_inputs1 = (outs1, noisy_labels)
        loss_inputs2 = (outs2, data2['noisy_labels'])

        if branch is None or branch in ['random']:
            losses = self.head.loss(*loss_inputs, reduction_override='none')
            losses1 = self.head.loss(*loss_inputs1, reduction_override='none')
            losses2 = self.head.loss(*loss_inputs2, reduction_override='none')
        elif branch in ['balance']:
            losses = self.head1.loss(*loss_inputs, reduction_override='none')
            losses1 = self.head1.loss(*loss_inputs1, reduction_override='none')
            losses2 = self.head1.loss(*loss_inputs2, reduction_override='none')

        losses['loss_cls'] = losses['loss_cls']*(union_labels > 0).float() + \
            (losses1['loss_cls']*(union_labels < 1).float() + \
             losses2['loss_cls']*(union_labels < 1).float())/2
        losses['loss_cls'] = losses['loss_cls'].mean()

        return losses

    def simple_test(self, img, img_meta, rescale=False, **kwargs):
        x = self.extract_feat(img)
        x1 = self.extract_feat(img, branch='balance')
        outs_random = self.head(x)
        outs_balance = self.head1(x1)
        factor = self.test_cfg.alpha
        outs = factor*outs_random + (1-factor)*outs_balance

        if self.savefeat:
            return outs, x

        if 'dual' in self.test_cfg and self.test_cfg.dual:
            return outs, outs_random, outs_balance
        return outs

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

