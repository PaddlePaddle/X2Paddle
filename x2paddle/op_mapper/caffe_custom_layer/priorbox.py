from .register import register
from x2paddle.core.util import *


def priorbox_shape(input_shape, max_size=None, aspect_ratio=None):
    fc_shape = input_shapes[0]
    N = 1
    if not max_size == None:
        N += 1
    if not aspect_ratio == None:
        N += 2 * len(aspect_ratio)
    N_bbx = fc_shape[2] * fc_shape[3] * N
    output_shape = [1, 2, 4 * N_bbx]
    return [output_shape]


def priorbox_layer(inputs,
                   step=0.0,
                   offset=0.5,
                   min_size=None,
                   max_size=None,
                   aspect_ratio=[1.0],
                   flip=False,
                   clip=False,
                   variance=[0.1, 0.1, 0.2, 0.2],
                   input_shape=None,
                   name=None):
    input = input_shape[0]
    image = input_shape[1]
    steps = tuple(step) if type(step) is list or type(step) is tuple else (step,
                                                                           step)
    box, variance_ = fluid.layers.prior_box(input,
                                            image,
                                            min_sizes=list(min_size),
                                            max_sizes=list(max_size),
                                            aspect_ratios=list(aspect_ratio),
                                            variance=list(variance),
                                            flip=flip,
                                            clip=clip,
                                            steps=step,
                                            offset=offset,
                                            name=name,
                                            min_max_aspect_ratios_order=True)
    box = fluid.layers.reshape(box, [1, 1, -1])
    variance_ = fluid.layers.reshape(variance_, [1, 1, -1])
    out = fluid.layers.concat([box, variance_], axis=1)
    return out


def priorbox_weights(name, data=None):
    weights_name = []
    return weights_name


register(kind='PriorBox',
         shape=priorbox_shape,
         layer=priorbox_layer,
         weights=priorbox_weights)
