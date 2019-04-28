#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:19:45 2019

@author: Macrobull
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from onnx2fluid.torch_export_helper import export_onnx_with_validation


# from https://github.com/santoshgsk/yolov2-pytorch/blob/master/yolotorch.ipynb
class Yolov2(nn.Module):
    def __init__(self):
        super(Yolov2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128,
                               out_channels=64,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.batchnorm5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256,
                               out_channels=128,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.batchnorm7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.batchnorm8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.batchnorm9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(in_channels=512,
                                out_channels=256,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        self.batchnorm10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(in_channels=256,
                                out_channels=512,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.batchnorm11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(in_channels=512,
                                out_channels=256,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        self.batchnorm12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(in_channels=256,
                                out_channels=512,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.batchnorm13 = nn.BatchNorm2d(512)

        self.conv14 = nn.Conv2d(in_channels=512,
                                out_channels=1024,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.batchnorm14 = nn.BatchNorm2d(1024)
        self.conv15 = nn.Conv2d(in_channels=1024,
                                out_channels=512,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        self.batchnorm15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(in_channels=512,
                                out_channels=1024,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.batchnorm16 = nn.BatchNorm2d(1024)
        self.conv17 = nn.Conv2d(in_channels=1024,
                                out_channels=512,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        self.batchnorm17 = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(in_channels=512,
                                out_channels=1024,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.batchnorm18 = nn.BatchNorm2d(1024)

        self.conv19 = nn.Conv2d(in_channels=1024,
                                out_channels=1024,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.batchnorm19 = nn.BatchNorm2d(1024)
        self.conv20 = nn.Conv2d(in_channels=1024,
                                out_channels=1024,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.batchnorm20 = nn.BatchNorm2d(1024)

        self.conv21 = nn.Conv2d(in_channels=3072,
                                out_channels=1024,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.batchnorm21 = nn.BatchNorm2d(1024)

        self.conv22 = nn.Conv2d(in_channels=1024,
                                out_channels=125,
                                kernel_size=1,
                                stride=1,
                                padding=0)

    def reorg_layer(self, x):
        stride = 2
        if hasattr(self, 'batch_size'):
            batch_size, channels, height, width = self.batch_size, self.channels, self.height, self.width
            new_ht = self.new_ht
            new_wd = self.new_wd
            new_channels = self.new_channels
        else:
            batch_size, channels, height, width = self.batch_size, self.channels, self.height, self.width = x.size(
            )
            new_ht = self.new_ht = height // stride
            new_wd = self.new_wd = width // stride
            new_channels = self.new_channels = channels * stride * stride

        passthrough = x.permute(0, 2, 3, 1)
        passthrough = passthrough.contiguous().view(-1, new_ht, stride, new_wd,
                                                    stride, channels)
        passthrough = passthrough.permute(0, 1, 3, 2, 4, 5)
        passthrough = passthrough.contiguous().view(-1, new_ht, new_wd,
                                                    new_channels)
        passthrough = passthrough.permute(0, 3, 1, 2)
        return passthrough

    def forward(self, x):
        out = F.max_pool2d(F.leaky_relu(self.batchnorm1(self.conv1(x)),
                                        negative_slope=0.1),
                           2,
                           stride=2)
        out = F.max_pool2d(F.leaky_relu(self.batchnorm2(self.conv2(out)),
                                        negative_slope=0.1),
                           2,
                           stride=2)

        out = F.leaky_relu(self.batchnorm3(self.conv3(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm4(self.conv4(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm5(self.conv5(out)), negative_slope=0.1)
        out = F.max_pool2d(out, 2, stride=2)

        out = F.leaky_relu(self.batchnorm6(self.conv6(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm7(self.conv7(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm8(self.conv8(out)), negative_slope=0.1)
        out = F.max_pool2d(out, 2, stride=2)

        out = F.leaky_relu(self.batchnorm9(self.conv9(out)), negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm10(self.conv10(out)),
                           negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm11(self.conv11(out)),
                           negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm12(self.conv12(out)),
                           negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm13(self.conv13(out)),
                           negative_slope=0.1)
        passthrough = self.reorg_layer(out)
        out = F.max_pool2d(out, 2, stride=2)

        out = F.leaky_relu(self.batchnorm14(self.conv14(out)),
                           negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm15(self.conv15(out)),
                           negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm16(self.conv16(out)),
                           negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm17(self.conv17(out)),
                           negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm18(self.conv18(out)),
                           negative_slope=0.1)

        out = F.leaky_relu(self.batchnorm19(self.conv19(out)),
                           negative_slope=0.1)
        out = F.leaky_relu(self.batchnorm20(self.conv20(out)),
                           negative_slope=0.1)

        out = torch.cat([passthrough, out], 1)
        out = F.leaky_relu(self.batchnorm21(self.conv21(out)),
                           negative_slope=0.1)
        out = self.conv22(out)

        return out


model = Yolov2()
model.eval()
xb = torch.rand((1, 3, 224, 224))
yp = model(xb)
export_onnx_with_validation(model, [xb],
                            'sample_yolov2', ['image'], ['pred'],
                            verbose=True,
                            training=False)
