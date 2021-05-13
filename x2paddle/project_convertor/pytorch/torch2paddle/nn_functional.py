# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import copy
from .utils import *


def binary_cross_entropy_with_logits(input,
                                     target,
                                     weight=None,
                                     size_average=None,
                                     reduce=None,
                                     reduction='mean',
                                     pos_weight=None):
    if not reduce or not size_average:
        reduction = "sum"
    input_t = str(input.dtype).lower().strip().split(".")[-1]
    if input_t in TYPE_MAPPER:
        input_t = TYPE_MAPPER[input_t]
    input_index = TYPE_ORDER.index(input_t)
    target_t = str(target.dtype).lower().strip().split(".")[-1]
    if target_t in TYPE_MAPPER:
        target_t = TYPE_MAPPER[target_t]
    target_index = TYPE_ORDER.index(target_t)
    if input_index < target_index:
        real_type = TYPE_ORDER[target_index]
        input = input.cast(real_type)
    else:
        real_type = TYPE_ORDER[input_index]
        target = target.cast(real_type)
    return paddle.nn.functional.binary_cross_entropy_with_logits(
        input, target, weight, reduction, pos_weight)


def avg_pool1d(input,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               count_include_pad=True):
    return paddle.nn.functional.avg_pool1d(
        input,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        exclusive=not count_include_pad)


def avg_pool2d(input,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               count_include_pad=True,
               divisor_override=None):
    return paddle.nn.functional.avg_pool2d(
        input,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        exclusive=not count_include_pad,
        divisor_override=divisor_override)


def avg_pool3d(input,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               count_include_pad=True,
               divisor_override=None):
    return paddle.nn.functional.avg_pool3d(
        input,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        exclusive=not count_include_pad,
        divisor_override=divisor_override)


def dropout(input, p=0.5, training=True, inplace=False):
    return paddle.nn.functional.dropout(input, p=p, training=training)


def interpolate(input,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=None,
                recompute_scale_factor=None):
    return paddle.nn.functional.interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners)


def leaky_relu(input, negative_slope=0.01, inplace=False):
    return paddle.nn.functional.leaky_relu(input, negative_slope=negative_slope)


def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    return paddle.nn.functional.log_softmax(input, axis=dim, dtype=None)


def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    paddle.nn.functional.mse_loss(input, target, reduction=reduction)


def relu(input, inplace=False):
    return paddle.nn.functional.relu(input)


def smooth_l1_loss(input,
                   target,
                   size_average=None,
                   reduce=None,
                   reduction='mean',
                   beta=1.0):
    paddle.nn.functional.smooth_l1_loss(
        input, target, reduction=reduction, delta=beta)


def softmax(input, dim=None, _stacklevel=3, dtype=None):
    return paddle.nn.functional.softmax(input, axis=dim, dtype=dtype)
