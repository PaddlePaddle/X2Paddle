from .register import register
from x2paddle.core.util import *
import numbers


def convolutiondepthwise_shape(input_shape,
                               num_output=None,
                               pad=None,
                               kernel_size=None,
                               stride=None,
                               dilation=None,
                               pad_h=None,
                               pad_w=None,
                               kernel_h=None,
                               kernel_w=None,
                               stride_h=None,
                               stride_w=None):
    [k_h, k_w] = [1, 1]
    if isinstance(kernel_size, numbers.Number):
        [k_h, k_w] = [kernel_size] * 2
    elif len(kernel_size) > 0:
        k_h = kernel_h if kernel_h else kernel_size[0]
        k_w = kernel_w if kernel_w else kernel_size[len(kernel_size) - 1]
    [s_h, s_w] = [1, 1]
    if isinstance(stride, numbers.Number):
        [s_h, s_w] = [stride] * 2
    elif len(stride) > 0:
        s_h = stride_h if stride_h else stride[0]
        s_w = stride_w if stride_w else stride[len(stride) - 1]
    [p_h, p_w] = [0, 0]
    if isinstance(pad, numbers.Number):
        [p_h, p_w] = [pad] * 2
    elif len(pad) > 0:
        p_h = pad_h if pad_h else pad[0]
        p_w = pad_w if pad_w else pad[len(pad) - 1]
    dila_len = len(dilation)
    dila_h = 1
    dila_w = 1
    if dila_len == 2:
        dila_h = dilation[0]
        dila_w = dilation[1]
    elif dila_len == 1:
        dila_h = dila_w = dilation[0]
    else:
        assert dila_len == 0, "invalid length[%s] of dilation in convolution" % (
            dila_len)
    i_w = input_shape[0][2]
    i_h = input_shape[0][3]
    o_h = (i_h + 2 * p_h - (dila_h * (k_h - 1) + 1)) / float(s_h) + 1
    o_w = (i_w + 2 * p_w - (dila_w * (k_w - 1) + 1)) / float(s_w) + 1
    import math
    o_h = int(math.floor(o_h))
    o_w = int(math.floor(o_w))
    c = num_output if num_output is not None else input_shape[0][1]
    return [[input_shape[0][0], c, o_h, o_w]]


def convolutiondepthwise_layer(inputs,
                               num_output=None,
                               pad=None,
                               kernel_size=None,
                               stride=None,
                               dilation=None,
                               pad_h=None,
                               pad_w=None,
                               kernel_h=None,
                               kernel_w=None,
                               stride_h=None,
                               stride_w=None,
                               input_shape=None,
                               name=None):
    import numbers
    [k_h, k_w] = [1, 1]
    if isinstance(kernel_size, numbers.Number):
        [k_h, k_w] = [kernel_size] * 2
    elif len(kernel_size) > 0:
        k_h = kernel_h if kernel_h else kernel_size[0]
        k_w = kernel_w if kernel_w else kernel_size[len(kernel_size) - 1]
    [s_h, s_w] = [1, 1]
    if isinstance(stride, numbers.Number):
        [s_h, s_w] = [stride] * 2
    elif len(stride) > 0:
        s_h = stride_h if stride_h else stride[0]
        s_w = stride_w if stride_w else stride[len(stride) - 1]
    [p_h, p_w] = [0, 0]
    if isinstance(pad, numbers.Number):
        [p_h, p_w] = [pad] * 2
    elif len(pad) > 0:
        p_h = pad_h if pad_h else pad[0]
        p_w = pad_w if pad_w else pad[len(pad) - 1]
    input = inputs[0]
    dila_len = len(dilation)
    dila_h = 1
    dila_w = 1
    if dila_len == 2:
        dila_h = dilation[0]
        dila_w = dilation[1]
    elif dila_len == 1:
        dila_h = dila_w = dilation[0]
    else:
        assert dila_len == 0, "invalid length[%s] of dilation in convolution" % (
            dila_len)
    c_in = input_shape[0][1]
    c_out = num_output if num_output is not None else input_shape[0][1]
    group = int(c_in / (c_in / c_out)) if c_in > c_out else int(c_in /
                                                                (c_out / c_in))
    out = fluid.layers.conv2d(input,
                              dilation=[dila_h, dila_w],
                              filter_size=[k_h, k_w],
                              stride=[s_h, s_w],
                              padding=[p_h, p_w],
                              groups=group,
                              num_filters=c_out,
                              param_attr=name + '_weights',
                              bias_attr=name + '_bias',
                              name=name)
    return out


def convolutiondepthwise_weights(name, data=None):
    weights_name = []
    weights_name.append(name + '_weights')
    weights_name.append(name + '_bias')
    return weights_name


register(kind='ConvolutionDepthwise',
         shape=convolutiondepthwise_shape,
         layer=convolutiondepthwise_layer,
         weights=convolutiondepthwise_weights)
