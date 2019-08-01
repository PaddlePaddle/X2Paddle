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
            pad_h = 0
            pad_w = 0
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


def shape_concat(layer, input_shape):
    params = layer.concat_param
    axis = params.axis
    output_shape = None
    for shape in input_shape:
        if output_shape is None:
            output_shape = shape
        else:
            output_shape[axis] += shape[axis]
    return [output_shape]


def shape_slice(layer, input_shape):
    inshape = input_shape[0]
    params = layer.slice_param
    axis = params.axis
    count = inshape[axis]
    points = list(params.slice_point)
    points = [0] + points + [count]
    output_shape = []
    for i in range(len(points)):
        shape = inshape
        size = points[i + 1] - points[i]
        shape[axis] = size
        output_shape.append(shape)
        if i == len(points) - 2:
            break
    return output_shape


def shape_prelu(layer, input_shape):
    return input_shape


def shape_sigmoid(layer, input_shape):
    return input_shape


def shape_absval(layer, input_shape):
    return input_shape


def shape_accuracy(layer, input_shape):
    return [[1]]


def shape_tanh(layer, input_shape):
    return input_shape


def shape_eltwise(layer, input_shape):
    return [input_shape[0]]


def shape_batchnorm(layer, input_shape):
    return input_shape


def shape_scale(layer, input_shape):
    return input_shape


def shape_reshape(layer, input_shape):
    def count(num_list):
        return reduce(lambda a, b: a * b, num_list)

    inshape = input_shape[0]
    params = layer.reshape_param
    axis = params.axis if hasattr(params, axis) else 0
    num_axes = params.num_axes if hasattr(params, num_axes) else -1
    if inshape[0] == -1:
        inshape[0] = 1
    input_count = count(inshape)

    input_num_axes = len(inshape)

    input_start_axis = axis
    start_axis = input_start_axis if input_start_axis >= 0 \
            else input_num_axes + input_start_axis + 1

    assert start_axis >= 0, "[Reshape]axis %d out of range" % (input_start_axis)
    assert start_axis <= input_num_axes, "[Reshape]axis %d out of range for %d-D input data"\
            % (input_start_axis, input_num_axes)

    assert num_axes >= -1, "[Reshape]num_axes must be >= 0, or -1 for all"

    end_axis = input_num_axes if num_axes == -1 else start_axis + num_axes
    assert end_axis <= input_num_axes, "end_axis[%d] = axis[%d] + num_axes[%d] is out of range"\
            % (end_axis, start_axis, num_axes)

    num_axes_replaced = end_axis - start_axis
    num_axes_retained = input_num_axes - num_axes_replaced
    num_new_axes = len(shape['dim'])
    outshape = []

    for i in range(start_axis):
        outshape.append(inshape[i])

    for i in range(num_new_axes):
        outshape.append(shape['dim'][i])

    for i in range(end_axis, input_num_axes):
        outshape.append(inshape[i])

    assert len(outshape) == num_axes_retained + num_new_axes,\
            "[Reshape]invalid dims of output shape[%s]" % (str(outshape))

    inferred_axis = -1
    copy_axes = []
    constant_count = 1
    for i in range(num_new_axes):
        top_dim = shape['dim'][i]
        if top_dim == 0:
            copy_axes.append(i)
            copy_axis_index = start_axis + i
            outshape[copy_axis_index] = inshape[copy_axis_index]
        elif top_dim == -1:
            assert inferred_axis == -1, "[Reshape]new shape contains multiple -1 dims"
            inferred_axis = i
        else:
            constant_count *= top_dim

    if inferred_axis >= 0:
        explicit_count = constant_count
        l = inshape[0:start_axis]
        if len(l) > 0:
            explicit_count *= count(l)

        l = inshape[end_axis:]
        if len(l) > 0:
            explicit_count *= count(l)

        for i in range(len(copy_axes)):
            explicit_count *= outshape[start_axis + copy_axes[i]]

        assert input_count % explicit_count == 0, "[Reshape]botom count[%d] "\
                "must be divisible by product of the specified dimensions[%d] "\
                % (input_count, explicit_count)
        outshape[start_axis + inferred_axis] = input_count / explicit_count

    output_count = count(outshape)
    assert output_count == input_count, "[Reshape]output count[%d] must match input count[%d]" % (
        output_count, input_count)
    if inshape[0] == -1:
        outshape[0] = -1
    return [outshape]


def shape_argmax(layer, input_shape):
    inshape = input_shape[0]
    params = layer.argmax_param
    out_max_val = params.out_max_val if hasattr(params, out_max_val) else False
    top_k = params.top_k if hasattr(params, top_k) else 1
    axis = parmas.axis if hasattr(params, axis) else -1
    if axis < 0:
        axis += len(inshape)
    assert (axis + 1 == len(inshape)
            ), 'only can be applied on the last dimension[axis:%d, %s] now,'\
                    'make sure you have set axis param in xxx.prototxt file' \
                    % (axis, str(inshape))

    outshape = inshape
    outshape[-1] = top_k
    if out_max_val is True:
        outshape[-1] *= 2
    return [outshape]


def shape_crop(layer, input_shape):
    assert len(input_shape) == 2, "the number of crop's inputs must be 2"
    return [input_shape[1]]


def shape_flatten(layer, input_shape):
    assert len(input_shape) == 1, "the number of flatten's inputs must be 1"
    params = layer.flatten_param
    start_axis = params.axis
    end_axis = params.end_axis
    if start_axis < 0:
        start_axis += len(input_shape[0])
    if end_axis < 0:
        end_axis += len(input_shape[0]) + 1
    assert start_axis <= end_axis, 'invalid axis[%d] or end_axis[%d] params'\
            % (start_axis, end_axis)
    output_shape = [0] * (start_axis - 0) + [
        -1
    ] + [0] * (len(input_shape[0]) - end_axis)
    return [output_shape]


def shape_power(layer, input_shape):
    return input_shape


def shape_reduction(layer, input_shape):
    params = layer.reduction_param
    axis = params.axis
    if axis < 0:
        axis += len(input_shape[0]) + 1
    assert axis <= len(input_shape[0]), 'invalid axis[%d] error' % (axis)
    return [input_shape[0:axis]]
