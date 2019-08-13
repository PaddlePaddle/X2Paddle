from .register import register
from x2paddle.core.util import *


def InstanceNormalization_shape(input_shape):
    return input_shape


def InstanceNormalization_layer(inputs, name=None):
    # TODO(lvmengsi@baidu.com): Check the accuracy when using fluid.layers.layer_norm.
    epsilon = 1e-5
    mean = fluid.layers.reduce_mean(inputs, dim=[2, 3], keep_dim=True)
    var = fluid.layers.reduce_mean(fluid.layers.square(inputs - mean),
                                   dim=[2, 3],
                                   keep_dim=True)
    if name is not None:
        scale_name = name + "_scale"
        offset_name = name + "_offset"
    scale_param = fluid.ParamAttr(name=scale_name,
                                  initializer=fluid.initializer.Constant(1.0),
                                  trainable=True)
    offset_param = fluid.ParamAttr(name=offset_name,
                                   initializer=fluid.initializer.Constant(0.0),
                                   trainable=True)
    scale = fluid.layers.create_parameter(attr=scale_param,
                                          shape=inputs.shape[1:2],
                                          dtype="float32")
    offset = fluid.layers.create_parameter(attr=offset_param,
                                           shape=inputs.shape[1:2],
                                           dtype="float32")

    tmp = fluid.layers.elementwise_mul(x=(inputs - mean), y=scale, axis=1)
    tmp = tmp / fluid.layers.sqrt(var + epsilon)
    tmp = fluid.layers.elementwise_add(tmp, offset, axis=1)
    return tmp


def InstanceNormalization_weights(name, data=None):
    weights_name = [name + '_scale']
    return weights_name


register(kind='InstanceNormalization',
         shape=InstanceNormalization_shape,
         layer=InstanceNormalization_layer,
         weights=InstanceNormalization_weights)
