from .register import register
from x2paddle.core.util import *


def roipooling_shape(input_shape, pooled_w=None, pooled_h=None):
    base_fea_shape = input_shapes[0]
    rois_shape = input_shapes[1]
    output_shape = base_fea_shape
    output_shape[0] = rois_shape[0]
    output_shape[2] = pooled_h
    output_shape[3] = pooled_w
    return [output_shape]


def roipooling_layer(inputs,
                     pooled_w=None,
                     pooled_h=None,
                     spatial_scale=None,
                     input_shape=None,
                     name=None):
    input = inputs[0]
    roi = inputs[1]
    roi = fluid.layers.slice(roi, axes=[1], starts=[1], ends=[5])
    out = fluid.layers.roi_pool(input,
                                roi,
                                pooled_height=pooled_h,
                                pooled_width=pooled_w,
                                spatial_scale=spatial_scale)
    return out


def roipooling_weights(name, data=None):
    weights_name = []
    return weights_name


register(kind='ROIPooling',
         shape=roipooling_shape,
         layer=roipooling_layer,
         weights=roipooling_weights)
