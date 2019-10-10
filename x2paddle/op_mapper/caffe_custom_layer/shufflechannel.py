from .register import register
from x2paddle.core.util import *


def shufflechannel_shape(input_shape):
    return input_shape


def shufflechannel_layer(inputs, group=None, input_shape=None, name=None):
    input = inputs[0]
    out = fluid.layers.shuffle_channel(x=input, group=group)
    return out


def shufflechannel_weights(name, data=None):
    weights_name = []
    return weights_name


register(kind='ShuffleChannel',
         shape=shufflechannel_shape,
         layer=shufflechannel_layer,
         weights=shufflechannel_weights)
