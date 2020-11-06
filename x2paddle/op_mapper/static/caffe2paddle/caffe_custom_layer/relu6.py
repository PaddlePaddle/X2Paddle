from .register import register
from x2paddle.core.util import *


def relu6_shape(input_shape):
    return input_shape


def relu6_layer(inputs, input_shape=None, name=None):
    input = inputs[0]
    out = fluid.layers.relu6(x=input)
    return out


def relu6_weights(name, data=None):
    weights_name = []
    return weights_name


register(
    kind='ReLU6', shape=relu6_shape, layer=relu6_layer, weights=relu6_weights)
