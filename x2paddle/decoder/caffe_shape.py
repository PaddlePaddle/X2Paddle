#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math


def get_params_w_h(params):
    if hasattr(params, 'dilation'):
        if len(params.dilation) == 0:
            dila_h = 1
            dila_w = 1
        elif len(params.dilation) == 1:
            dila_h = params.dilation[0]
            dila_w = params.dilation[0]
        else:
            dila_h = params.dilation[0]
            dila_w = params.dilation[1]
    else:
        dila_h = 1
        dila_w = 1

    if not isinstance(getattr(params, 'pad'), int):
        if len(params.pad) == 0:
            pad_h = 1
            pad_w = 1
        elif len(params.pad) == 1:
            pad_h = params.pad[0]
            pad_w = params.pad[0]
        else:
            pad_h, pad_w, = params.pad[0]
            pad_w = params.pad[1]
        if params.pad_h != 0 or params.pad_w != 0:
            pad_h = params.pad_h
            pad_w = params.pad_w
    else:
        if params.pad_h != 0 or params.pad_w != 0:
            pad_h = params.pad_h
            pad_w = params.pad_w
        else:
            pad_h = getattr(params, 'pad')
            pad_w = getattr(params, 'pad')

    if not isinstance(getattr(params, 'kernel_size'), int):
        if len(params.kernel_size) == 0:
            kernel_h = 1
            kernel_w = 1
        elif len(params.kernel_size) == 1:
            kernel_h = params.kernel_size[0]
            kernel_w = params.kernel_size[0]
        else:
            kernel_h = params.kernel_size[0]
            kernel_w = params.kernel_size[1]
        if params.kernel_h != 0 or params.kernel_w != 0:
            kernel_h = params.kernel_h
            kernel_w = params.kernel_w
    else:
        if params.kernel_h != 0 or params.kernel_w != 0:
            kernel_h = params.kernel_h
            kernel_w = params.kernel_w
        else:
            kernel_h = getattr(params, 'kernel_size')
            kernel_w = getattr(params, 'kernel_size')
    if not isinstance(getattr(params, 'stride'), int):
        if len(params.stride) == 0:
            stride_h = 1
            stride_w = 1
        elif len(params.stride) == 1:
            stride_h = params.stride[0]
            stride_w = params.stride[0]
        else:
            stride_h = params.stride[0]
            stride_w = params.stride[1]
        if params.stride_h != 0 or params.stride_w != 0:
            stride_h = params.stride_h
            stride_w = params.stride_w
    else:
        if params.stride_h != 0 or params.stride_w != 0:
            stride_h = params.stride_h
            stride_w = params.stride_w
        else:
            stride_h = getattr(params, 'stride')
            stride_w = getattr(params, 'stride')
    return dila_h, dila_w, pad_h, pad_w, kernel_h, kernel_w, stride_h, stride_w


def get_filter_output_shape(i_h, i_w, params, round_func):
    dila_h, dila_w, pad_h, pad_w, kernel_h, kernel_w, stride_h, stride_w = get_params_w_h(
        params)
    o_h = (i_h + 2 * pad_h - (dila_h *
                              (kernel_h - 1) + 1)) / float(stride_h) + 1
    o_w = (i_w + 2 * pad_w - (dila_w *
                              (kernel_w - 1) + 1)) / float(stride_w) + 1

    return (int(round_func(o_h)), int(round_func(o_w)))


def get_strided_kernel_output_shape(params, input_shape, round_func):

    o_h, o_w = get_filter_output_shape(input_shape[2], input_shape[3], params,
                                       round_func)
    has_c_o = hasattr(params, 'num_output')
    c = params.num_output if has_c_o else input_shape[1]
    return [[input_shape[0], c, o_h, o_w]]


def shape_convolution(layer, input_shape):
    params = layer.convolution_param
    return get_strided_kernel_output_shape(params, input_shape[0], math.floor)


def shape_deconvolution(layer, input_shape):
    h_i = input_shape[2]
    w_i = input_shape[3]

    params = layer.convolution_param
    dila_h, dila_w, pad_h, pad_w, kernel_h, kernel_w, stride_h, stride_w = get_params_w_h(
        params)

    h_o = (h_i - 1) * stride_h - 2 * pad_h + dila_h * (kernel_h - 1) + 1
    w_o = (w_i - 1) * stride_w - 2 * pad_w + dila_w * (kernel_w - 1) + 1

    has_c_o = hasattr(params, 'num_output')
    c = params.num_output if has_c_o else input_shape.channels
    return [[input_shape[0][0], c, h_o, w_o]]


def shape_pooling(layer, input_shape):
    params = layer.pooling_param
    global_pool = getattr(params, 'global_pooling', False)
    if global_pool:
        return [[input_shape[0][0], input_shape[0][1], 1, 1]]

    ceil_mode = getattr(params, 'ceil_mode', True)
    if ceil_mode is True:
        method = math.ceil
    else:
        method = math.floor
    return get_strided_kernel_output_shape(params, input_shape[0], method)


def shape_innerproduct(layer, input_shape):
    params = layer.inner_product_param
    return [[input_shape[0][0], params.num_output]]


def shape_lrn(layer, input_shape):
    return input_shape


def shape_relu(layer, input_shape):
    return input_shape


def shape_softmax(layer, input_shape):
    return input_shape


def shape_input(layer, input_shape):
    return [list(layer.input_param.shape[0].dim)]
