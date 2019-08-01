from .register import register
from x2paddle.core.util import *


def permute_shape(input_shape, order=None):
    inshape = input_shape[0]
    output_shape = []
    for ii in order:
        assert ii < len(inshape), "invalid order for permute[%s]" % (name)
        output_shape.append(inshape[ii])
    return [output_shape]


def permute_layer(inputs, order=None, input_shape=None, name=None):
    input = inputs[0]
    order = list(order)
    out = fluid.layers.transpose(input, perm=order, name=name)
    return out


def permute_weights(name, data=None):
    weights_name = []
    return weights_name


register(kind='Permute',
         shape=permute_shape,
         layer=permute_layer,
         weights=permute_weights)
