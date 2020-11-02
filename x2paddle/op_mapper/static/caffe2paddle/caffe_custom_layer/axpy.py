from .register import register
from x2paddle.core.util import *


def axpy_shape(input_shapes):
    assert len(input_shapes) == 3, "not valid input shape for axpy layer"
    assert len(input_shapes[0]) == len(input_shapes[1]), 'should have same dims'
    output_shape = input_shapes[1]
    assert (input_shapes[2] == output_shape),\
            "shape not consistent for axpy[%s <--> %s]" \
            % (str(output_shape), str(input_shapes[2]))
    return [output_shape]


def axpy_layer(inputs, input_shape=None, name=None):
    alpha = inputs[0]
    x = inputs[1]
    y = inputs[2]
    out = fluid.layers.elementwise_mul(x, alpha, axis=0)
    out = fluid.layers.elementwise_add(out, y, name=name)
    return out


def axpy_weights(name, data=None):
    weights_name = []
    return weights_name


register(kind='Axpy', shape=axpy_shape, layer=axpy_layer, weights=axpy_weights)
