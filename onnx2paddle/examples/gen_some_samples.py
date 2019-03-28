#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:19:45 2019

@author: Macrobull

Not all ops in this file are supported by both Pytorch and ONNX
This only demostrates the conversion/validation workflow from Pytorch to ONNX to Paddle

"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from onnx2paddle.torch_export_helper import export_onnx_with_validation


idx = 0


######### example: RNN ########
#
#class Model(nn.Module):
#    def __init__(self):
#        super(Model, self).__init__()
#        self.rnn = nn.RNN(4, 6, 2)
#
#    def forward(self, x):
#        y = x
#        y, h = self.rnn(y)
#        return y
#
#
#model = Model()
#xb = torch.rand((2, 3, 4))
#yp = model(xb)
#idx += 1
#print('index: ', idx)
#export_onnx_with_validation(model, (xb, ), 't' + str(idx),
#                            ['x'], ['y'],
#                            verbose=True, training=False)


######### example: random ########
#
#class Model(nn.Module):
#    def __init__(self):
#        super(Model, self).__init__()
#
#    def forward(self, x):
#        y = torch.rand((2, 3)) # + torch.rand_like(xb)
#        y = y + torch.randn((2, 3)) # + torch.randn_like(xb)
#        return y
#
#
#model = Model()
#xb = torch.rand((2, 3))
#yp = model(xb)
#idx += 1
#print('index: ', idx)
#export_onnx_with_validation(model, (xb, ), 't' + str(idx),
#                            ['x'], ['y'],
#                            verbose=True, training=False)


######## example: fc ########

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(3, 8)

    def forward(self, x):
        y = x
        y = self.fc(y)
        return y


model = Model()
xb = torch.rand((2, 3))
yp = model(xb)
idx += 1
print('index: ', idx)
export_onnx_with_validation(model, (xb, ), 't' + str(idx),
                            ['x'], ['y'],
                            verbose=True, training=False)


######## example: compare ########

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x0, x1):
        x0 = x0.clamp(-1, 1)
        a = torch.max(x0, x1) == x1
        b = x0 < x1
        c = x0 > x1
        return a, b, c


model = Model()
xb0 = torch.rand((2, 3))
xb1 = torch.rand((2, 3))
ya, yb, yc = model(xb0, xb1)
idx += 1
print('index: ', idx)
export_onnx_with_validation(model, (xb0, xb1), 't' + str(idx),
                            ['x0', 'x1'], ['ya', 'yb', 'yc'],
                            verbose=True, training=False)

######## example: affine_grid ########

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, theta):
        grid = F.affine_grid(theta, (2, 2, 8, 8))
        return grid


model = Model()
theta = torch.rand((2, 2, 3))
grid = model(theta)
idx += 1
print('index: ', idx)
export_onnx_with_validation(model, (theta, ), 't' + str(idx),
                            ['theta'], ['grid'],
                            verbose=True, training=False)


######## example: conv2d_transpose ########

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.ConvTranspose2d(3, 8, 3)
        self.dropout = nn.Dropout2d()

    def forward(self, x):
        y = x
        y = self.conv(y)
        y = self.dropout(y)
        return y


model = Model()
xb = torch.rand((2, 3, 4, 5))
yp = model(xb)
idx += 1
print('index: ', idx)
export_onnx_with_validation(model, (xb, ), 't' + str(idx),
                            ['x'], ['y'],
                            verbose=True, training=False)

######## example: conv2d ########

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.batch_norm = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(2)

    def forward(self, x):
        y = x
        y = self.conv(y)
        y = self.batch_norm(y)
        y = self.pool(y)
        return y


model = Model()
xb = torch.rand((2, 3, 4, 5))
yp = model(xb)
idx += 1
print('index: ', idx)
export_onnx_with_validation(model, (xb, ), 't' + str(idx),
                            ['x'], ['y'],
                            verbose=True, training=False)


######### example: conv1d ########
#
#class Model(nn.Module):
#    def __init__(self):
#        super(Model, self).__init__()
#        self.batch_norm = nn.BatchNorm2d(3)
#
#    def forward(self, x):
#        y = x
#        y = self.batch_norm(y)
#        return y
#
#
#model = Model()
#xb = torch.rand((2, 3, 4, 5))
#yp = model(xb)
#idx += 1
#print('index: ', idx)
#export_onnx_with_validation(model, (xb, ), 't' + str(idx),
#                            ['x'], ['y'],
#                            verbose=True, training=False)

######## example: empty ########

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x


model = Model()
xb = torch.rand((2, 3))
yp = model(xb)
idx += 1
print('index: ', idx)
export_onnx_with_validation(model, (xb, ), 't' + str(idx),
                            ['y'], ['y'],
                            verbose=True, training=False)
