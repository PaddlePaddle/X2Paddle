from .register import register
from x2paddle.core.util import *


def shufflechannel_shape(input_shape):
    return input_shape


def shufflechannel_layer(inputs, group=None, input_shape=None, name=None):
    input = inputs[0]
    c_fm = fluid.layers.split(input, num_or_sections=input_shape[0][1], dim=1)
    size = int(input_shape[0][1]/group)
    new_c_fm = []
    for i in range(size):
        for j in range(group):
            new_c_fm.append(c_fm[j * size + i])
    out = fluid.layers.concat(new_c_fm, axis = 1)
    return out


def shufflechannel_weights(name, data=None):
    weights_name = []
    return weights_name


register(kind='ShuffleChannel',
         shape=shufflechannel_shape,
         layer=shufflechannel_layer,
         weights=shufflechannel_weights)
