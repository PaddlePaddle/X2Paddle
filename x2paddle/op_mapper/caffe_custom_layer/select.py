from .register import register
from x2paddle.core.util import *


def select_shape(input_shape, axis=None, slice_point=None):
    inshape = input_shape[0]
    slice_point = slice_point
    start = slice_point[0]
    if len(slice_point) == 2:
        end = slice_point[1]
    else:
        end = input_shape[axis]
    assert end > start, "invalid slice_point with [start:%d, end:%d]" % (start,
                                                                         end)
    output_shape = input_shape
    output_shape[axis] = end - start
    return [output_shape]


def select_layer(inputs,
                 axis=None,
                 slice_point=None,
                 input_shape=None,
                 name=None):
    input = inputs[0]
    maxint32 = 2147483647
    slice_point = [0] + slice_point
    slice_point.append(maxint32)
    i = 0
    out = []
    for i in range(len(slice_point)):
        out.append(
            fluid.layers.slice(input,
                               axes=[axis],
                               starts=[slice_point[i]],
                               ends=[slice_point[i + 1]],
                               name=name + '_' + str(i)))
        if i == len(slice_point) - 2:
            break
    return out


def select_weights(name, data=None):
    weights_name = []
    return weights_name


register(kind='Select',
         shape=select_shape,
         layer=select_layer,
         weights=select_weights)
