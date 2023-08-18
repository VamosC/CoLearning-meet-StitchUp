from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch
import torchvision
from ..registry import NECKS


@NECKS.register_module
class PFC(nn.Module):

    def __init__(self, in_channels, out_channels, dropout, bn=True, norm=False,relu=False,layers=1):
        super(PFC, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.run_bn = bn
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.fc_hidden_1 = nn.Linear(out_channels, out_channels)
        self.norm = norm
        self.dropout = dropout
        self.layers = layers
        self.avg_pool = nn.AvgPool2d(in_channels)
        self.relu= nn.LeakyReLU(0.2, inplace=True) if relu else None

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

    def init_weights(self):
        init.kaiming_normal_(self.fc.weight, mode='fan_out')
        init.constant_(self.fc.bias, 0)
        init.constant_(self.bn.weight, 1)
        init.constant_(self.bn.bias, 0)
        init.constant_(self.bn2.weight, 1)
        init.constant_(self.bn2.bias, 0)

    def forward(self, x, need_feat=False, x2=None, use_bn2=False):
        if isinstance(x, tuple):
            x = x[-1]
            if x2 is not None:
                x2 = x2[-1]
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        feat = self.fc(x)
        if x2 is not None:
            x2 = F.avg_pool2d(x2, x2.size()[2:])
            x2 = x2.view(x2.size(0), -1)
            feat2 = self.fc(x2)

        if self.run_bn:
            if x2 is None:
                if use_bn2:
                    x = self.bn2(feat)
                else:
                    x = self.bn(feat)
            else:
                x = self.bn(feat)
                x2 = self.bn2(feat2)
        else:
            x = feat

        if self.dropout > 0:
            x = self.drop(x)

        if x2 is not None:
            if self.dropout > 0:
                x2 = self.drop(x2)
            neck_feat1 = x
            neck_feat2 = x2
            x = (x+x2)/2

        if self.layers == 2:
            x = self.fc_hidden_1(x)
            x = self.bn(x)
            if self.dropout > 0:
                x = self.drop(x)

        if self.relu is not None:
            x = self.relu(x)

        if need_feat:
            return x, neck_feat1, neck_feat2
        return x

