# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: Drift
Email:  wutuobang@baidu.com
Date:   2020/04/22 18:45
"""

from .register import register
from x2paddle.core.util import *


def upsample_shape(input_shapes, scale):
    """

    :param input_shapes:
    :param scale:
    :return:
    """
    assert len(input_shapes) == 1, "not valid input shape for upsample layer"
    assert type(scale) is int

    input_shape = input_shapes[0]
    new_h = scale * input_shape[2]
    new_w = scale * input_shape[3]

    output_shape = [input_shape[0], input_shape[1], new_h, new_w]
    return [output_shape]


def upsample_layer(inputs, scale, input_shape=None, name=None):
    """

    :param inputs:
    :param scale:
    :param input_shape:
    :param name:
    :return:
    """
    x = inputs[0]
    out = fluid.layers.resize_nearest(
        x, align_corners=False, scale=scale, name=name)

    return out


def upsample_weights(name, data=None):
    """

    :param name:
    :param data:
    :return:
    """
    weights_name = []
    return weights_name


register(
    kind='Upsample',
    shape=upsample_shape,
    layer=upsample_layer,
    weights=upsample_weights)
