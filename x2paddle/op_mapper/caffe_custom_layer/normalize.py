from .register import register
from x2paddle.core.util import *


def normalize_shape(input_shape):
    return input_shape


def normalize_layer(inputs,
                    across_spatial=None,
                    channel_shared=None,
                    input_shape=None,
                    name=None):
    assert across_spatial == False, "Only support across_spatial == False for Normalize"
    input = inputs[0]
    l2_norm = fluid.layers.l2_normalize(input, axis=1, name=name + '_l2')
    scale_param = fluid.layers.create_parameter(
        shape=[1] if channel_shared else [1, 1, 1, input_shape[0][1]],
        dtype=input.dtype,
        attr=name + '_scale')
    scale_param = fluid.layers.reshape(x=scale_param, \
                  shape=[1] if channel_shared else [input_shape[0][1]])
    out = fluid.layers.elementwise_mul(x=l2_norm,
                                       y=scale_param,
                                       axis=-1 if channel_shared else 1)
    return out


def normalize_weights(name, data=None):
    weights_name = [name + '_scale']
    return weights_name


register(kind='Normalize',
         shape=normalize_shape,
         layer=normalize_layer,
         weights=normalize_weights)
