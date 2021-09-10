# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import numbers
from functools import reduce


def get_kernel_parameters(params):
    [k_h, k_w] = [1, 1]
    if isinstance(params.kernel_size, numbers.Number):
        [k_h, k_w] = [params.kernel_size] * 2
    elif len(params.kernel_size) > 0:
        k_h = params.kernel_h if params.kernel_h > 0 else params.kernel_size[0]
        k_w = params.kernel_w if params.kernel_w > 0 else params.kernel_size[
            len(params.kernel_size) - 1]
    elif params.kernel_h > 0 or params.kernel_w > 0:
        k_h = params.kernel_h
        k_w = params.kernel_w
    [s_h, s_w] = [1, 1]
    if isinstance(params.stride, numbers.Number):
        [s_h, s_w] = [params.stride] * 2
    elif len(params.stride) > 0:
        s_h = params.stride_h if params.stride_h > 0 else params.stride[0]
        s_w = params.stride_w if params.stride_w > 0 else params.stride[len(
            params.stride) - 1]
    elif params.stride_h > 0 or params.stride_w > 0:
        s_h = params.stride_h
        s_w = params.stride_w
    [p_h, p_w] = [0, 0]
    if isinstance(params.pad, numbers.Number):
        [p_h, p_w] = [params.pad] * 2
    elif len(params.pad) > 0:
        p_h = params.pad_h if params.pad_h > 0 else params.pad[0]
        p_w = params.pad_w if params.pad_w > 0 else params.pad[len(params.pad) -
                                                               1]
    elif params.pad_h > 0 or params.pad_w > 0:
        p_h = params.pad_h
        p_w = params.pad_w
    dila_h = dila_w = 1
    if hasattr(params, 'dilation'):
        dila_len = len(params.dilation)
        if dila_len == 2:
            dila_h = params.dilation[0]
            dila_w = params.dilation[1]
        elif dila_len == 1:
            dila_h = dila_w = params.dilation[0]
        else:
            assert dila_len == 0, "invalid length[%s] of dilation in convolution" % (
                dila_len)
    return dila_h, dila_w, p_h, p_w, k_h, k_w, s_h, s_w


def get_strided_kernel_output_shape(params, input_shape, round_func):
    i_h = input_shape[2]
    i_w = input_shape[3]
    dila_h, dila_w, pad_h, pad_w, kernel_h, kernel_w, stride_h, stride_w = get_kernel_parameters(
        params)
    o_h = (i_h + 2 * pad_h - (dila_h *
                              (kernel_h - 1) + 1)) / float(stride_h) + 1
    o_w = (i_w + 2 * pad_w - (dila_w *
                              (kernel_w - 1) + 1)) / float(stride_w) + 1
    o_h = int(round_func(o_h))
    o_w = int(round_func(o_w))
    has_c_o = hasattr(params, 'num_output')
    c = params.num_output if has_c_o else input_shape[1]
    return [[input_shape[0], c, o_h, o_w]]


def shape_convolution(layer, input_shape):
    params = layer.convolution_param
    return get_strided_kernel_output_shape(params, input_shape[0], math.floor)


def shape_depthwiseconvolution(layer, input_shape):
    return shape_convolution(layer, input_shape)


def shape_deconvolution(layer, input_shape):

    h_i = input_shape[0][2]
    w_i = input_shape[0][3]

    params = layer.convolution_param
    dila_h, dila_w, pad_h, pad_w, kernel_h, kernel_w, stride_h, stride_w = get_kernel_parameters(
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
    if not hasattr(params, 'ceil_mode'):
        round_mode = getattr(params, 'round_mode', 0)
        if round_mode == 1:
            method = math.floor
        else:
            method = math.ceil
    return get_strided_kernel_output_shape(params, input_shape[0], method)


def shape_convolutiondepthwise(layer, input_shape):
    params = layer.convolution_param
    return get_strided_kernel_output_shape(params, input_shape[0], math.floor)


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


def shape_memorydata(layer, input_shape):
    params = layer.memory_data_param
    shape = []
    shape.append(int(params.batch_size))
    shape.append(int(params.channels))
    shape.append(int(params.height))
    shape.append(int(params.width))
    return [shape]


def shape_concat(layer, input_shape):
    params = layer.concat_param
    axis = params.axis
    output_shape = None
    for shape in input_shape:
        if output_shape is None:
            output_shape = []
            for i in range(len(shape)):
                output_shape.append(shape[i])
        else:
            output_shape[axis] += shape[axis]
    return [output_shape]


def shape_slice(layer, input_shape):
    inshape = input_shape[0]

    top_len = len(layer.top)
    params = layer.slice_param
    axis = params.axis
    slice_dim = params.slice_dim
    if slice_dim != 1 and axis == 1:
        axis = slice_dim
    points = list(params.slice_point)
    count = inshape[axis]
    if len(points) == 0:
        assert count % top_len == 0, "the parameter of Slice is wrong"
        part = count / top_len
        t = part
        while t < count:
            points.append(int(t))
            t += part
    points = [0] + points + [count]
    output_shape = []
    for i in range(len(points)):
        shape = []
        for ii in range(len(inshape)):
            shape.append(inshape[ii])
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
    axis = params.axis if hasattr(params, 'axis') else 0
    num_axes = params.num_axes if hasattr(params, 'num_axes') else -1
    is_unknow_batch = False
    if inshape[0] == -1:
        is_unknow_batch = True
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
    num_new_axes = len(list(params.shape.dim))
    output_shape = []

    for i in range(start_axis):
        output_shape.append(inshape[i])

    for i in range(num_new_axes):
        output_shape.append(params.shape.dim[i])

    for i in range(end_axis, input_num_axes):
        output_shape.append(inshape[i])

    assert len(output_shape) == num_axes_retained + num_new_axes,\
            "[Reshape]invalid dims of output shape[%s]" % (str(output_shape))

    inferred_axis = -1
    copy_axes = []
    constant_count = 1
    for i in range(num_new_axes):
        top_dim = params.shape.dim[i]
        if top_dim == 0:
            copy_axes.append(i)
            copy_axis_index = start_axis + i
            output_shape[copy_axis_index] = inshape[copy_axis_index]
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
            explicit_count *= output_shape[start_axis + copy_axes[i]]
        assert input_count % explicit_count == 0, "[Reshape]botom count[%d] "\
                "must be divisible by product of the specified dimensions[%d] "\
                % (input_count, explicit_count)
        output_shape[start_axis + inferred_axis] = int(input_count /
                                                       explicit_count)

    output_count = count(output_shape)
    assert output_count == input_count, "[Reshape]output count[%d] must match input count[%d]" % (
        output_count, input_count)
    if is_unknow_batch:
        output_shape[0] = -1
    return [output_shape]


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

    output_shape = inshape
    output_shape[-1] = top_k
    if out_max_val is True:
        output_shape[-1] *= 2
    return [output_shape]


def shape_crop(layer, input_shape):
    assert len(input_shape) == 2, "the number of crop's inputs must be 2"
    return [input_shape[1]]


def shape_flatten(layer, input_shape):
    assert len(input_shape) == 1, "the number of flatten's inputs must be 1"
    inshape = input_shape[0]
    params = layer.flatten_param
    start_axis = params.axis
    end_axis = params.end_axis
    if start_axis < 0:
        start_axis += len(inshape)
    if end_axis < 0:
        end_axis += len(inshape) + 1
    assert start_axis <= end_axis, 'invalid axis[%d] or end_axis[%d] params'\
            % (start_axis, end_axis)
    output_shape = inshape[0:start_axis]
    if len(inshape[start_axis:end_axis]) != 0:
        flat_sz = reduce(lambda a, b: a * b, inshape[start_axis:end_axis])
        output_shape += [flat_sz]
    output_shape += inshape[end_axis:len(inshape)]
    output_shape[0] = -1
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


def shape_axpy(layer, input_shapes):
    assert len(input_shapes) == 3, "not valid input shape for axpy layer"
    assert len(input_shapes[0]) == len(input_shapes[1]), 'should have same dims'
    output_shape = input_shapes[1]
    assert (input_shapes[2] == output_shape),\
            "shape not consistent for axpy[%s <--> %s]" \
            % (str(output_shape), str(input_shapes[2]))
    return [output_shape]


def shape_detectionoutput(layer, input_shape):
    return [[-1, 6]]


def shape_normalize(layer, input_shape):
    return input_shape


def shape_permute(layer, input_shape):
    order = layer.permute_param.order
    inshape = input_shape[0]
    output_shape = []
    order = list(order)
    for ii in order:
        assert ii < len(inshape), "invalid order for permute[%s]" % (name)
        output_shape.append(inshape[ii])
    return [output_shape]


def shape_priorbox(layer, input_shape):
    max_size = layer.prior_box_param.max_size
    aspect_ratio = layer.prior_box_param.aspect_ratio
    fc_shape = input_shape[0]
    N = 1
    if not max_size == None:
        N += 1
    if not aspect_ratio == None:
        N += 2 * len(aspect_ratio)
    N_bbx = fc_shape[2] * fc_shape[3] * N
    output_shape = [1, 2, 4 * N_bbx]
    return [output_shape]


def shape_relu6(layer, input_shape):
    return input_shape


def shape_roipooling(layer, input_shapes):
    pooled_w = layer.roi_pooling_param.pooled_w
    pooled_h = layer.roi_pooling_param.pooled_h
    base_fea_shape = input_shapes[0]
    rois_shape = input_shapes[1]
    output_shape = base_fea_shape
    output_shape[0] = rois_shape[0]
    output_shape[2] = pooled_h
    output_shape[3] = pooled_w
    return [output_shape]


def shape_shufflechannel(layer, input_shape):
    return input_shape


def shape_upsample(layer, input_shapes):
    scale = layer.upsample_param.scale
    assert len(input_shapes) == 1, "not valid input shape for upsample layer"
    assert type(scale) is int
    input_shape = input_shapes[0]
    new_h = scale * input_shape[2]
    new_w = scale * input_shape[3]

    output_shape = [input_shape[0], input_shape[1], new_h, new_w]
    return [output_shape]


def shape_select(layer, input_shapes):
    slice_point = layer.select_param.slice_point
    axis = layer.select_param.axis
    input_shape = input_shapes[0]
    start = slice_point[0]
    if len(slice_point) == 2:
        end = slice_point[1]
    else:
        end = input_shape[axis]
    assert end > start, "invalid slice_point with [start:%d, end:%d]"\
             % (start, end)
    output_shape = input_shape
    output_shape[axis] = end - start
    return [output_shape]
