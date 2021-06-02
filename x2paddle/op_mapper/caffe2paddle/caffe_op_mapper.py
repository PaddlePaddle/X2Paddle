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

import sys
import numbers
import numpy as np
from x2paddle.core.util import *
from x2paddle.core.program import PaddleGraph
from x2paddle.decoder.caffe_decoder import CaffeGraphNode


def _adjust_parameters(node):
    data = node.data
    # When using the protobuf-backend, each parameter initially has four dimensions.
    # In certain cases (like FC layers), we want to eliminate the singleton dimensions.
    # This implementation takes care of the common cases. However, it does leave the
    # potential for future issues.
    # The Caffe-backend does not suffer from this problem.
    data = list(data)

    squeeze_indices = [1]  # Squeeze biases.
    if node.layer_type == 'InnerProduct':
        squeeze_indices.append(0)  # Squeeze FC.

    for idx in squeeze_indices:
        if idx >= len(data):
            continue

        d = data[idx]
        assert len(
            d.shape
        ) == 4, 'invalid shape[%s] from caffe when adjust_parameters' % (
            str(d.shape))

        shape_old = d.shape
        sq_axis = None
        if idx == 0:
            sq_axis = (0, 1)
        elif idx == 1:
            sq_axis = (0, 1, 2)
        else:
            continue

        data[idx] = np.squeeze(d, axis=sq_axis)
        shape_new = data[idx].shape
    return data


def _get_kernel_parameters(kind, params):
    assert kind in [
        "Convolution", "Pooling", "Deconvolution", "ConvolutionDepthwise"
    ]
    [k_h, k_w] = [1, 1]
    if params.kernel_h > 0 or params.kernel_w > 0:
        k_h = params.kernel_h
        k_w = params.kernel_w
    elif isinstance(params.kernel_size, numbers.Number):
        [k_h, k_w] = [params.kernel_size] * 2
    elif len(params.kernel_size) > 0:
        k_h = params.kernel_h if params.kernel_h > 0 else params.kernel_size[0]
        k_w = params.kernel_w if params.kernel_w > 0 else params.kernel_size[
            len(params.kernel_size) - 1]
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
    group = 1
    c_o = 1
    if kind in ["Convolution", "Deconvolution", "ConvolutionDepthwise"]:
        if kind in ["Convolution", "Deconvolution"]:
            c_o = params.num_output
        dila_len = len(params.dilation)
        if dila_len == 2:
            dila_h = params.dilation[0]
            dila_w = params.dilation[1]
        elif dila_len == 1:
            dila_h = dila_w = params.dilation[0]
        else:
            assert dila_len == 0, "invalid length[%s] of dilation in convolution" % (
                dila_len)
    if kind in ['Convolution', 'Deconvolution']:
        group = params.group
    kernel = [k_h, k_w]
    stride = [s_h, s_w]
    pad = [p_h, p_w]
    dilation = [dila_h, dila_w]
    return c_o, kernel, stride, pad, dilation, group


class CaffeOpMapper():
    directly_map_ops = {
        'Sigmoid': ['paddle.nn.layer.Sigmoid'],
        'TanH': ['paddle.nn.Tanh'],
    }

    def __init__(self, decoder):
        self.graph = decoder.caffe_graph
        if not self.op_checker():
            raise Exception("Model is not supported yet.")
        self.params = dict()
        self.paddle_graph = PaddleGraph(parent_layer=None, source_type="caffe")
        self.paddle_graph.outputs = self.graph.output_nodes
        self.input_index = 0
        self.inputs_info = {}
        self.nn_name2id = {}
        print("Total nodes: {}".format(
            sum([
                isinstance(node, CaffeGraphNode)
                for name, node in self.graph.node_map.items()
            ])))
        print("Nodes converting ...")
        for i, node_name in enumerate(self.graph.topo_sort):
            sys.stderr.write("\rConverting node {} ...     ".format(i + 1))
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if hasattr(self, op):
                func = getattr(self, op)
                func(node)
            elif op in self.directly_map_ops:
                self.directly_map(node)
        print("\nNodes converted.")
        self.paddle_graph.set_name(self.graph.graph_name)
        self.paddle_graph.set_parameters(self.params)
        self.paddle_graph.set_inputs_info(self.inputs_info)

    def op_checker(self):
        unsupported_ops = set()
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if not hasattr(self, op) and op not in self.directly_map_ops:
                unsupported_ops.add(op)
        if len(unsupported_ops) == 0:
            return True
        else:
            if len(unsupported_ops) > 0:
                print("\n========= {} OPs are not supported yet ===========".
                      format(len(unsupported_ops)))
            for op in unsupported_ops:
                print("========== {} ============".format(op))
            return False

    def directly_map(self, node):
        inputs = node.layer.input
        assert len(inputs) == 1, 'directly_map error with multi inputs'
        op_info = self.directly_map_ops[node.layer_type]
        input = self.graph.get_input_node(node, 0)
        paddle_op = op_info[0]
        if paddle_op.startswith("paddle.nn"):
            op_name = paddle_op[10:].lower()
            op_name = name_generator(op_name, self.nn_name2id)
            output_name = node.name
            layer_outputs = [op_name, output_name]
            self.paddle_graph.add_layer(
                kernel=paddle_op,
                inputs={"x": input.name},
                outputs=layer_outputs)
        else:
            self.paddle_graph.add_layer(
                kernel=paddle_op, inputs={"x": input.name},
                outputs=[node.name])

    def Input(self, node):
        self.paddle_graph.add_layer(
            "paddle.to_tensor",
            inputs={},
            outputs=[node.layer_name],
            data="x{}".format(self.input_index))
        shape = list(node.layer.input_param.shape[0].dim)[1:]
        self.inputs_info["x{}".format(self.input_index)] = [[-1] + shape,
                                                            "float32"]
        self.input_index += 1

    def MemoryData(self, node):
        params = node.layer.memory_data_param
        transform_params = node.layer.transform_param
        self.paddle_graph.add_layer(
            "paddle.to_tensor",
            inputs={},
            outputs=[node.layer_name],
            data="x{}".format(self.input_index))
        shape = list()
        shape.append(params.batch_size)
        shape.append(params.channels)
        if hasattr(transform_params, "crop_size"):
            shape.append(transform_params.crop_size)
            shape.append(transform_params.crop_size)
        else:
            shape.append(params.width)
            shape.append(params.height)
        self.inputs_info["x{}".format(self.input_index)] = [shape, "float32"]
        self.input_index += 1

    def Convolution(self, node):
        conv2d_name = name_generator("conv", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [conv2d_name, output_name]
        data = node.data
        params = node.layer.convolution_param
        out_channel, kernel, stride, pad, dilation, group = _get_kernel_parameters(
            node.layer_type, params)
        if data is None:
            data = []
            print(
                "The parameter of {} (type is {}) is not set. So we set the parameters as 0"
                .format(node.layer_name, node.layer_type))
            data.append(
                np.zeros([
                    out_channel, node.in_shapes[0][1], kernel[0], kernel[1]
                ]).astype('float32'))
            data.append(np.zeros([out_channel, ]).astype('float32'))
        else:
            data = _adjust_parameters(node)
        self.params[conv2d_name + ".weight"] = data[0]
        if len(data) == 2:
            self.params[conv2d_name + ".bias"] = data[1]
        assert len(node.inputs
                   ) == 1, "The count of Convolution node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        layer_attrs = {
            "in_channels": node.in_shapes[0][1],
            "out_channels": out_channel,
            "kernel_size": kernel,
            "stride": stride,
            "padding": pad,
            "dilation": dilation,
            "groups": group
        }
        if len(data) == 1:
            layer_attrs["bias_attr"] = False
        self.paddle_graph.add_layer(
            "paddle.nn.Conv2D",
            inputs={"input": input.name},
            outputs=layer_outputs,
            **layer_attrs)

    def DepthwiseConvolution(self, node):
        node.layer_type = "ConvolutionDepthwise"
        self.ConvolutionDepthwise(node)

    def Deconvolution(self, node):
        conv2d_name = name_generator("conv", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [conv2d_name, output_name]
        data = node.data
        params = node.layer.convolution_param
        out_channel, kernel, stride, pad, dilation, group = _get_kernel_parameters(
            node.layer_type, params)
        if data is None:
            data = []
            print(
                "The parameter of {} (type is {}) is not set. So we set the parameters as 0"
                .format(node.layer_name, node.layer_type))
            data.append(
                np.zeros([
                    out_channel, node.in_shapes[0][1], kernel[0], kernel[1]
                ]).astype('float32'))
            data.append(np.zeros([out_channel, ]).astype('float32'))
        else:
            data = _adjust_parameters(node)
        self.params[conv2d_name + ".weight"] = data[0]
        if len(data) == 2:
            self.params[conv2d_name + ".bias"] = data[1]
        assert len(node.inputs
                   ) == 1, "The count of Deconvolution node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        layer_attrs = {
            "in_channels": node.in_shapes[0][1],
            "out_channels": out_channel,
            "kernel_size": kernel,
            "stride": stride,
            "padding": pad,
            "dilation": dilation,
            "groups": group
        }
        if len(data) == 1:
            layer_attrs["bias_attr"] = False
        self.paddle_graph.add_layer(
            "paddle.nn.Conv2DTranspose",
            inputs={"input": input.name},
            outputs=layer_outputs,
            **layer_attrs)

    def ConvolutionDepthwise(self, node):
        conv2d_name = name_generator("conv", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [conv2d_name, output_name]
        data = node.data
        params = node.layer.convolution_param
        out_channel, kernel, stride, pad, dilation, group = _get_kernel_parameters(
            node.layer_type, params)
        out_channel = params.num_output if params.num_output is not None else node.in_shapes[
            0][1]
        in_channel = node.in_shapes[0][1]
        group = int(in_channel / (
            in_channel / out_channel)) if in_channel > out_channel else int(
                in_channel / (out_channel / in_channel))
        if data is None:
            data = []
            print(
                "The parameter of {} (type is {}) is not set. So we set the parameters as 0"
                .format(node.layer_name, node.layer_type))
            data.append(
                np.zeros([
                    out_channel, node.in_shapes[0][1], kernel[0], kernel[1]
                ]).astype('float32'))
            data.append(np.zeros([out_channel, ]).astype('float32'))
        else:
            data = _adjust_parameters(node)
        self.params[conv2d_name + ".weight"] = data[0]
        if len(data) == 2:
            self.params[conv2d_name + ".bias"] = data[1]
        assert len(node.inputs
                   ) == 1, "The count of Deconvolution node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        layer_attrs = {
            "in_channels": in_channel,
            "out_channels": out_channel,
            "kernel_size": kernel,
            "stride": stride,
            "padding": pad,
            "dilation": dilation,
            "groups": group
        }
        if len(data) == 1:
            layer_attrs["bias_attr"] = False
        self.paddle_graph.add_layer(
            "paddle.nn.Conv2D",
            inputs={"input": input.name},
            outputs=layer_outputs,
            **layer_attrs)

    def Pooling(self, node):
        pool2d_name = name_generator("pool", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [pool2d_name, output_name]
        params = node.layer.pooling_param
        ceil_mode = getattr(params, "ceil_mode", True)
        if not hasattr(params, 'ceil_mode'):
            ceil_mode = True if getattr(params, "round_mode", 0) == 0 else False
        global_pool = getattr(params, "global_pooling", False)
        kernel_default = [1, 1]
        channel, kernel, stride, pad, dilation, group = _get_kernel_parameters(
            node.layer_type, params)
        if params.pool == 0:
            pool_type = "max"
        else:
            pool_type = "avg"
        assert len(
            node.inputs) == 1, "The count of Pooling node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        if global_pool:
            if kernel[0] == 0:
                kernel = [1, 1]
            if params.pool == 0:
                self.paddle_graph.add_layer(
                    "paddle.nn.AdaptiveMaxPool2D",
                    inputs={"input": input.name},
                    outputs=layer_outputs,
                    output_size=kernel)
            else:
                self.paddle_graph.add_layer(
                    "paddle.nn.AdaptiveAvgPool2D",
                    inputs={"input": input.name},
                    outputs=layer_outputs,
                    output_size=kernel)
        else:
            layer_attrs = {
                'kernel_size': kernel,
                'stride': stride,
                'padding': pad,
                'ceil_mode': ceil_mode,
            }
            if params.pool == 0:
                self.paddle_graph.add_layer(
                    "paddle.nn.MaxPool2D",
                    inputs={"input": input.name},
                    outputs=layer_outputs,
                    **layer_attrs)
            else:
                self.paddle_graph.add_layer(
                    "paddle.nn.AvgPool2D",
                    inputs={"input": input.name},
                    outputs=layer_outputs,
                    **layer_attrs)

    def LRN(self, node):
        lrn_name = name_generator("lrn", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [lrn_name, output_name]
        assert len(node.inputs) == 1, "The count of LRN node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.lrn_param
        assert params.local_size % 2 == 1
        alpha = params.alpha / float(params.local_size)
        layer_attrs = {
            "n": params.local_size,
            "k": params.k,
            "alpha": alpha,
            "beta": params.beta,
        }
        self.paddle_graph.add_layer(
            "paddle.fluid.layers.lrn",
            inputs={"input": input.name},
            outputs=[node.layer_name],
            **layer_attrs)

    def InnerProduct(self, node):
        linear_name = name_generator("linear", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [linear_name, output_name]
        data = node.data
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.inner_product_param
        if data is None:
            print(
                "The parameter of {} (type is {}) is not set. So we set the parameters as 0."
                .format(node.layer_name, node.layer_type))
            data = []
            data.append(
                np.zeros([node.in_shapes[0][1], params.num_output]).astype(
                    "float32").astype("float32"))
            data.append(
                np.zeros([params.num_output]).astype("float32").astype(
                    "float32"))
        else:
            data = _adjust_parameters(node)
            # Reshape the parameters to Paddle's ordering
            transpose_order = (1, 0)
            w = data[0]
            fc_shape = w.shape
            output_channels = fc_shape[0]
            w = w.reshape((output_channels, -1))
            w = w.transpose(transpose_order)
            data[0] = w

        self.params[linear_name + ".weight"] = data[0]
        if len(data) == 2:
            self.params[linear_name + ".bias"] = data[1]
        assert len(node.inputs
                   ) == 1, "The count of InnerProduct node\'s input is not 1."
        assert params.axis == 1
        assert params.bias_term == True
        layer_attrs = {
            "in_features": data[0].shape[0],
            "out_features": params.num_output
        }
        if len(data) == 1:
            layer_attrs["bias"] = False
        if node.in_shapes[0][-1] != data[0].shape[0]:
            self.paddle_graph.add_layer(
                "paddle.reshape",
                inputs={"x": input.name},
                outputs=[output_name],
                shape=[-1, data[0].shape[0]])
            self.paddle_graph.add_layer(
                "paddle.nn.Linear",
                inputs={"input": output_name},
                outputs=layer_outputs,
                **layer_attrs)
        else:
            self.paddle_graph.add_layer(
                "paddle.nn.Linear",
                inputs={"input": input.name},
                outputs=layer_outputs,
                **layer_attrs)

    def AbsVal(self, node):
        assert len(
            node.inputs
        ) >= 1, "The count of AbsVal node\'s input is not more than 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        self.paddle_graph.add_layer(
            "paddle.abs",
            inputs={"input": input.name},
            outputs=[node.layer_name])

    def Softmax(self, node):
        softmax_name = name_generator("softmax", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [softmax_name, output_name]
        assert len(
            node.inputs) == 1, "The count of Softmax node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.softmax_param
        axis = params.axis
        shape = node.in_shapes[0]
        dims = len(shape)
        axis = axis + dims if axis < 0 else axis
        layer_attrs = {'axis': axis}
        self.paddle_graph.add_layer(
            "paddle.nn.Softmax",
            inputs={"input": input.name},
            outputs=layer_outputs,
            **layer_attrs)

    def Slice(self, node):
        assert len(
            node.inputs) == 1, "The count of Slice node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        top_len = len(node.layer.top)
        params = node.layer.slice_param
        axis = params.axis
        slice_dim = params.slice_dim
        if slice_dim != 1 and axis == 1:
            axis = slice_dim
        output_shape = node.out_shapes
        sections_list = list()
        outputs_list = list()
        for i, s in enumerate(output_shape):
            sections_list.append(s[axis])
            outputs_list.append("{}_p{}".format(node.layer_name, i))
        layer_attrs = {
            'num_or_sections': sections_list,
            'axis': axis,
        }
        self.paddle_graph.add_layer(
            "paddle.split",
            inputs={"x": input.name},
            outputs=outputs_list,
            **layer_attrs)

    def Concat(self, node):
        assert len(
            node.inputs
        ) >= 1, "The count of Concat node\'s input is not more than 1."
        inputs_list = list()
        for i in range(len(node.inputs)):
            input = self.graph.get_input_node(node, idx=i, copy=True)
            inputs_list.append(input.name)
        params = node.layer.concat_param
        axis = params.axis
        layer_attrs = {'axis': axis}
        self.paddle_graph.add_layer(
            "paddle.concat",
            inputs={"x": inputs_list},
            outputs=[node.layer_name],
            **layer_attrs)

    def ReLU(self, node):
        relu_name = name_generator("relu", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [relu_name, output_name]
        assert len(
            node.inputs) == 1, "The count of RelU node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.relu_param
        if params.HasField('negative_slope') and params.negative_slope != 0:
            negative_slope = float(params.negative_slope)

            layer_attrs = {'negative_slope': negative_slope}
            self.paddle_graph.add_layer(
                "paddle.nn.LeakyReLU",
                inputs={"input": input.name},
                outputs=layer_outputs,
                **layer_attrs)
        else:
            self.paddle_graph.add_layer(
                "paddle.nn.ReLU",
                inputs={"input": input.name},
                outputs=layer_outputs)

    def PReLU(self, node):
        prelu_name = name_generator("prelu", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [prelu_name, output_name]
        assert len(
            node.inputs) == 1, "The count of PReLU node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.prelu_param
        mode_bool = params.channel_shared
        output_shape = node.out_shapes[0]
        if mode_bool:
            num_parameters = 1
        else:
            num_parameters = output_shape[1]
        data = node.data
        self.params[prelu_name + '._weight'] = np.squeeze(data[0])
        assert data is not None, "The parameter of {} (type is {}) is not set. You need to use python package of caffe to set the default value.".format(
            node.layer_name, node.layer_type)
        self.paddle_graph.add_layer(
            "paddle.nn.PReLU",
            inputs={"input": input.name},
            outputs=layer_outputs,
            num_parameters=num_parameters)

    def Eltwise(self, node):
        assert len(
            node.inputs) == 2, "The count of Eltwise node\'s input is not 2."
        params = node.layer.eltwise_param
        mode = params.operation
        inputs = []
        input0 = self.graph.get_input_node(node, idx=0, copy=True)
        input1 = self.graph.get_input_node(node, idx=1, copy=True)
        input0_name = input0.name
        input1_name = input1.name
        if mode == 0:
            inputs_dict = {}
            inputs_dict['x'] = input0_name
            inputs_dict['y'] = input1_name
            self.paddle_graph.add_layer(
                "paddle.multiply",
                inputs=inputs_dict,
                outputs=[node.layer_name])
        elif mode == 1:
            if hasattr(params, 'coeff') and len(params.coeff) == 2:
                coeff = params.coeff
                self.paddle_graph.add_layer(
                    "paddle.scale",
                    inputs={"x": input0_name},
                    outputs=[node.layer_name + '_mul0'],
                    scale=coeff[0])
                self.paddle_graph.add_layer(
                    "paddle.scale",
                    inputs={"x": input1_name},
                    outputs=[node.layer_name + '_mul1'],
                    scale=coeff[1])
                inputs_dict = {}
                inputs_dict['x'] = node.layer_name + '_mul0'
                inputs_dict['y'] = node.layer_name + '_mul1'
                self.paddle_graph.add_layer(
                    "paddle.add", inputs=inputs_dict,
                    outputs=[node.layer_name])
            else:
                inputs_dict = {}
                inputs_dict['x'] = input0_name
                inputs_dict['y'] = input1_name
                self.paddle_graph.add_layer(
                    "paddle.add", inputs=inputs_dict,
                    outputs=[node.layer_name])
        else:
            inputs_dict = {}
            inputs_dict['x'] = input0_name
            inputs_dict['y'] = input1_name
            self.paddle_graph.add_layer(
                "paddle.max", inputs=inputs_dict, outputs=[node.layer_name])

    def BatchNorm(self, node):
        batchnorm_name = name_generator("batchnorm", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [batchnorm_name, output_name]
        assert len(
            node.inputs) == 1, "The count of BatchNorm node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.batch_norm_param
        if hasattr(params, "eps"):
            eps = params.eps
        else:
            eps = 1e-5
        if node.data is None or len(node.data) != 3:
            print(
                "The parameter of {} (type is {}) is not set. So we set the parameters as 0"
                .format(node.layer_name, node.layer_type))
            mean = np.zeros([node.in_shapes[0][1], ]).astype("float32")
            variance = np.zeros([node.in_shapes[0][1], ]).astype("float32")
            scale = 0
        else:

            node.data = [np.squeeze(i).astype("float32") for i in node.data]
            mean, variance, scale = node.data
        # Prescale the stats
        scaling_factor = 1.0 / scale if scale != 0 else 0
        mean *= scaling_factor
        variance *= scaling_factor
        self.params[batchnorm_name + "._mean"] = mean
        self.params[batchnorm_name + '._variance'] = variance
        layer_attrs = {
            "num_features": node.in_shapes[0][1],
            "epsilon": eps,
            "weight_attr": False,
            "bias_attr": False,
        }
        if len(node.in_shapes[0]) == 2:
            self.paddle_graph.add_layer(
                "paddle.unsqueeze",
                inputs={"x": input.name},
                outputs=[input.name],
                axis=[2, 3])
        self.paddle_graph.add_layer(
            "paddle.nn.BatchNorm2D",
            inputs={"input": input.name},
            outputs=layer_outputs,
            **layer_attrs)
        if len(node.in_shapes[0]) == 2:
            self.paddle_graph.add_layer(
                "paddle.squeeze",
                inputs={"x": node.layer_name},
                outputs=[node.layer_name],
                axis=[2, 3])

    def Scale(self, node):
        if node.data is None:
            print(
                "The parameter of {} (type is {}) is not set. So we set the parameters as 0"
                .format(node.layer_name, node.layer_type))
            self.params[node.layer_name + "_cparam1"] = np.zeros([
                node.in_shapes[0][1],
            ]).astype("float32")
            self.params[node.layer_name + "_cparam2"] = np.zeros([
                node.in_shapes[0][1],
            ]).astype("float32")
        else:
            self.params[node.layer_name + "_cparam1"] = np.squeeze(node.data[
                0]).astype("float32")
            if not node.layer.scale_param.bias_term:
                self.params[node.layer_name + "_cparam2"] = np.zeros([
                    node.in_shapes[0][1],
                ]).astype("float32")
            else:
                self.params[node.layer_name + "_cparam2"] = np.squeeze(
                    node.data[1]).astype("float32")
        params = node.layer.scale_param
        axis = params.axis
        inputs = []
        if len(node.inputs) == 2:
            input0 = self.graph.get_input_node(node, idx=0, copy=True)
            input1 = self.graph.get_input_node(node, idx=1, copy=True)
            input0_name = input0.name
            input1_name = input1.name
            inputs_dict = {}
            inputs_dict['x'] = input0_name
            inputs_dict['y'] = input1_name
            self.paddle_graph.add_layer(
                "paddle.multiply",
                inputs=inputs_dict,
                outputs=[node.layer_name + "_mul"],
                axis=1)
        else:
            self.paddle_graph.add_layer(
                "self.create_parameter",
                inputs={},
                outputs=[node.layer_name + "_cparam1"],
                shape=self.params[node.layer_name + "_cparam1"].shape,
                attr=string(node.layer_name + "_cparam1"))
            input0 = self.graph.get_input_node(node, idx=0, copy=True)
            input0_name = input0.name
            inputs_dict = {}
            inputs_dict['x'] = input0_name
            inputs_dict['y'] = node.layer_name + "_cparam1"
            if len(node.in_shapes[0]) == 2:
                self.paddle_graph.add_layer(
                    "paddle.multiply",
                    inputs=inputs_dict,
                    outputs=[node.layer_name + "_mul"])
            else:
                self.paddle_graph.add_layer(
                    "paddle.multiply",
                    inputs=inputs_dict,
                    outputs=[node.layer_name + "_mul"],
                    axis=axis)
        self.paddle_graph.add_layer(
            "self.create_parameter",
            inputs={},
            outputs=[node.layer_name + "_cparam2"],
            shape=self.params[node.layer_name + "_cparam2"].shape,
            attr=string(node.layer_name + "_cparam2"))
        inputs_dict = {}
        inputs_dict['x'] = node.layer_name + "_mul"
        inputs_dict['y'] = node.layer_name + "_cparam2"
        output_shape = node.out_shapes[0]
        if axis == -1:
            self.paddle_graph.add_layer(
                "paddle.add", inputs=inputs_dict, outputs=[node.layer_name])
        else:
            if axis < 0:
                axis = axis + len(output_shape)
            param2_shape = self.params[node.layer_name + "_cparam2"].shape
            param2_shape_len = len(param2_shape)
            diff_len = len(output_shape) - axis - param2_shape_len
            new_shape = list(param2_shape) + [1] * diff_len
            self.paddle_graph.add_layer(
                "paddle.reshape",
                inputs={"x": node.layer_name + "_cparam2"},
                outputs=[node.layer_name + "_cparam2"],
                shape=new_shape)
            self.paddle_graph.add_layer(
                "paddle.add", inputs=inputs_dict, outputs=[node.layer_name])

    def Reshape(self, node):
        input = self.graph.get_input_node(node, idx=0, copy=True)
        output_shape = node.out_shapes[0]
        self.paddle_graph.add_layer(
            "paddle.reshape",
            inputs={"x": input.name},
            outputs=[node.layer_name],
            shape=output_shape)

    def ArgMax(self, node):
        assert len(node.inputs) == 1 and len(
            node.outputs
        ) == 1, "The count of ArgMax node\'s input and output is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        input_shape = node.in_shapes[0]
        params = node.layer.argmax_param
        out_max_val = params.out_max_val if hasattr(params,
                                                    out_max_val) else False
        top_k = params.top_k if hasattr(params, top_k) else 1
        axis = params.axis if hasattr(params, axis) else -1
        if axis < 0:
            axis += len(input_shape)
        if out_max_val is True:
            self.paddle_graph.add_layer(
                "paddle.topk",
                inputs={"x": input.name},
                outputs=[
                    node.layer_name + "_topk_var",
                    node.layer_name + "_index_var"
                ],
                k=top_k)
            self.paddle_graph.add_layer(
                "paddle.cast",
                inputs={"x": node.layer_name + "_index_var"},
                outputs=[node.layer_name + "_index_var"],
                dtype="{}_topk_var.dtype".format(node.layer_name))
            self.paddle_graph.add_layer(
                "paddle.concat",
                inputs={
                    "x": [
                        node.layer_name + "_topk_var",
                        node.layer_name + "_index_var"
                    ]
                },
                outputs=[node.layer_name],
                axis=axis)
        else:
            self.paddle_graph.add_layer(
                "paddle.topk",
                inputs={"x": input.name},
                outputs=["_", node.layer_name],
                k=top_k)

    def Axpy(self, node):
        assert len(node.inputs) == 1 and len(
            node.outputs
        ) == 1, "The count of Axpy node\'s input and output is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.axpy_param
        input0 = self.graph.get_input_node(node, idx=0, copy=True)
        input1 = self.graph.get_input_node(node, idx=1, copy=True)
        input2 = self.graph.get_input_node(node, idx=2, copy=True)
        input0_name = input0.name
        input1_name = input1.name
        input2_name = input2.name
        inputs_dict = {}
        inputs_dict['x'] = input1_name
        inputs_dict['y'] = input0_name
        self.paddle_graph.add_layer(
            "paddle.multiply",
            inputs=inputs_dict,
            outputs=[node.layer_name + "_mul"],
            axis=0)
        inputs_dict = {}
        inputs_dict['x'] = node.layer_name + "_mul"
        inputs_dict['y'] = input2_name
        self.paddle_graph.add_layer(
            "paddle.add",
            inputs=inputs_dict,
            outputs=[node.layer_name + "_mul"])

    def Crop(self, node):
        assert len(
            node.inputs) == 2, "The count of Crop node\'s input is not 2."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        example = self.graph.get_input_node(node, idx=1, copy=True)
        params = node.layer.crop_param
        axis = params.axis
        input_shape = node.in_shapes[0]
        if axis < 0:
            axis += len(input_shape)
        offset_real = [0] * len(input_shape)
        if hasattr(params, "offset") and len(params.offset) > 0:
            offset = list(params.offset)
            assert (len(input_shape) - axis
                    ) == len(offset), "invalid offset[%s] in crop layer" % (
                        str(offset))
            offset_real = [0] * axis + offset
        self.paddle_graph.add_layer(
            "paddle.crop",
            inputs={"x": input.name},
            outputs=[node.layer_name],
            shape=node.in_shapes[1],
            offsets=list(offset_real))

    def Flatten(self, node):
        assert len(
            node.
            inputs) == 1, "The count of DetectionOutput node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        self.paddle_graph.add_layer(
            "paddle.reshape",
            inputs={"x": input.name},
            outputs=[node.layer_name],
            shape=node.out_shapes[0])

    def Power(self, node):
        assert len(
            node.inputs) == 1, "The count of Permute node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.power_param
        layer_attrs = {
            'scale': params.scale,
            'bias': params.shift,
            'bias_after_scale': True
        }
        self.paddle_graph.add_layer(
            "paddle.scale",
            inputs={"x": input.name},
            outputs=[node.layer_name],
            **layer_attrs)
        self.paddle_graph.add_layer(
            "paddle.pow",
            inputs={"x": node.layer_name},
            outputs=[node.layer_name],
            exponent=params.power)

    def Reduction(self, node):
        assert len(
            node.inputs) == 1, "The count of Reduction node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.reduction_param
        operation = params.operation
        axis = params.axis
        coeff = params.coeff
        assert operation >= 1 and operation <= 4, "reduction reduction [%s] error" % (
            operation)
        input_len = len(node.in_shapes[0])
        if axis < 0:
            axis += input_len + 1
        dim = list(range(input_len))
        # operation = SUM
        if operation == 1:
            layer_attrs = {
                "dim": dim[axis:],
                "keep_dim": False,
            }
            self.paddle_graph.add_layer(
                "paddle.sum",
                inputs={"input": input.name},
                outputs=[node.layer_name],
                **layer_attrs)
        # operation = ASUM
        elif operation == 2:
            self.paddle_graph.add_layer(
                "paddle.abs",
                inputs={"x": input.name},
                outputs=[node.layer_name])
            layer_attrs = {
                "dim": dim[axis:],
                "keep_dim": False,
            }
            self.paddle_graph.add_layer(
                "paddle.sum",
                inputs={"input": node.layer_name},
                outputs=[node.layer_name],
                **layer_attrs)
        # operation = SUMSQ
        elif operation == 3:
            self.paddle_graph.add_layer(
                "paddle.pow",
                inputs={"x": input.name},
                outputs=[node.layer_name],
                exponent=2.0)
            layer_attrs = {
                "dim": dim[axis:],
                "keep_dim": False,
            }
            self.paddle_graph.add_layer(
                "paddle.sum",
                inputs={"input": node.layer_name},
                outputs=[node.layer_name],
                **layer_attrs)
        # operation = MEAN
        else:
            layer_attrs = {
                "axis": dim[axis:],
                "keepdim": False,
            }
            self.paddle_graph.add_layer(
                "paddle.mean",
                inputs={"x": input.name},
                outputs=[node.layer_name],
                **layer_attrs)
        self.paddle_graph.add_layer(
            "paddle.scale",
            inputs={"x": node.layer_name},
            outputs=[node.layer_name],
            scale=coeff)

    def DetectionOutput(self, node):
        detection_output_name = name_generator("detection_output",
                                               self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [detection_output_name, output_name]
        assert len(
            node.
            inputs) == 3, "The count of DetectionOutput node\'s input is not 3."
        inputs_dict = dict()
        for i in range(len(node.inputs)):
            input = self.graph.get_input_node(node, idx=i, copy=True)
            if i == 1:
                input = self.graph.get_input_node(node, idx=i, copy=True)
                while input is not None \
                      and input.layer_type != 'Softmax' \
                      and input.layer_type != 'Sigmoid':
                    input = self.graph.get_input_node(input, idx=0, copy=True)
                assert input is not None, 'This kind of DetectionOutput is not supported!'
                input = self.graph.get_input_node(input, idx=0, copy=True)
            inputs_dict["x{}".format(i)] = input.name
        params = node.layer.detection_output_param
        nms_param = params.nms_param
        nms_param_dict = dict()
        nms_param_dict["nms_threshold"] = nms_param.nms_threshold
        nms_param_dict["top_k"] = nms_param.top_k
        nms_param_dict["eta"] = nms_param.eta
        if nms_param is None:
            nms_param_dict = {"nms_threshold": 0.3, "top_k": 10, "eta": 1.0}
        default = {"nms_threshold": 0.3, "top_k": 10, "eta": 1.0}
        fields = ["eta", "top_k", "nms_threshold"]
        for f in default.keys():
            if f not in nms_param_dict:
                nms_param_dict[f] = default[f]
        layer_attrs = {
            "background_label": params.background_label_id,
            "nms_threshold": nms_param_dict["nms_threshold"],
            "nms_top_k": nms_param_dict["top_k"],
            "keep_top_k": params.keep_top_k,
            "score_threshold": params.confidence_threshold,
            "nms_eta": nms_param_dict["eta"]
        }
        self.paddle_graph.add_layer(
            kernel="custom_layer:DetectionOutput",
            inputs=inputs_dict,
            outputs=layer_outputs,
            **layer_attrs)

    def Normalize(self, node):
        normalize_name = name_generator("normalize", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [normalize_name, output_name]
        assert len(
            node.inputs) == 1, "The count of Normalize node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.norm_param
        param_name = node.layer_name + "_scale"
        if node.data is None or len(node.data) != 1:
            print(
                "The parameter of {} (type is {}) is not set. So we set the parameters as 0"
                .format(node.layer_name, node.layer_type))
            self.params[param_name] = \
                np.zeros([1] if params.channel_shared else [node.in_shapes[0][1]]).astype("float32")
        else:
            self.params[param_name] = _adjust_parameters(node)[0]

        self.paddle_graph.add_layer(
            "self.create_parameter",
            inputs={},
            outputs=[param_name],
            shape=self.params[param_name].shape,
            attr=string(param_name))
        inputs_dict = {}
        layer_attrs = {"axis": -1 if params.channel_shared else 1}
        self.paddle_graph.add_layer(
            "custom_layer:Normalize",
            inputs={"x": input.name,
                    "param": param_name},
            outputs=layer_outputs,
            **layer_attrs)

    def Permute(self, node):
        assert len(
            node.inputs) == 1, "The count of Permute node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.permute_param
        order = list(params.order)
        self.paddle_graph.add_layer(
            "paddle.transpose",
            inputs={"x": input.name},
            outputs=[node.layer_name],
            perm=order)

    def PriorBox(self, node):
        priorbox_name = name_generator("priorbox", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [priorbox_name, output_name]
        assert len(
            node.inputs) == 2, "The count of PriorBox node\'s input is not 2."
        input0 = self.graph.get_input_node(node, idx=0, copy=True)
        input1 = self.graph.get_input_node(node, idx=1, copy=True)
        inputs_dict = {}
        inputs_dict["x0"] = input0.name
        inputs_dict["x1"] = input1.name
        params = node.layer.prior_box_param
        steps = tuple(params.step) if type(params.step) \
                is list or type(params.step) is tuple \
                else (params.step, params.step)
        layer_attrs = {
            "min_sizes": params.min_size,
            "max_sizes": params.max_size,
            "aspect_ratios": params.aspect_ratio,
            "variance": params.variance,
            "flip": params.flip,
            "clip": params.clip,
            "steps": steps,
            "offset": params.offset,
            "min_max_aspect_ratios_order": True
        }
        self.paddle_graph.add_layer(
            "custom_layer:PriorBox",
            inputs=inputs_dict,
            outputs=layer_outputs,
            **layer_attrs)

    def ReLU6(self, node):
        relu6_name = name_generator("relu6", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [relu6_name, output_name]
        assert len(
            node.inputs) == 1, "The count of RelU6 node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        self.paddle_graph.add_layer(
            "paddle.nn.ReLU6",
            inputs={"input": input.name},
            outputs=layer_outputs)

    def ROIPooling(self, node):
        roipooling_name = name_generator("roipooling", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [roipooling_name, output_name]
        assert len(
            node.inputs) == 2, "The count of ROIPooling node\'s input is not 2."
        input0 = self.graph.get_input_node(node, idx=0, copy=True)
        input1 = self.graph.get_input_node(node, idx=1, copy=True)
        inputs_dict = {}
        inputs_dict["x0"] = input0.name
        inputs_dict["x1"] = input1.name
        params = node.layer.roi_pooling_param
        layer_attrs = {
            "pooled_height": params.pooled_h,
            "pooled_width": params.pooled_w,
            "spatial_scale": params.spatial_scale
        }
        self.paddle_graph.add_layer(
            "custom_layer:ROIPooling",
            inputs=inputs_dict,
            outputs=layer_outputs,
            **layer_attrs)

    def ShuffleChannel(self, node):
        assert len(node.inputs
                   ) == 1, "The count of ShuffleChannel node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.shuffle_channel_param
        self.paddle_graph.add_layer(
            "paddle.fluid.layers.shuffle_channel",
            inputs={"x": input.name},
            outputs=[node.layer_name],
            group=params.group)

    def Upsample(self, node):
        assert len(
            node.inputs) == 1, "The count of Upsample node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        params = node.layer.upsample_param
        layer_attrs = {
            "align_corners": False,
            "scale_factor": params.scale,
            "mode": "nearest"
        }
        self.paddle_graph.add_layer(
            "paddle.nn.functional.interpolate",
            inputs={"x": input.name},
            outputs=[node.layer_name],
            **layer_attrs)

    def Select(self, node):
        select_name = name_generator("select", self.nn_name2id)
        output_name = node.layer_name
        layer_outputs = [select_name, output_name]
        assert len(
            node.inputs) == 1, "The count of Select node\'s input is not 1."
        input = self.graph.get_input_node(node, idx=0, copy=True)
        input_shape = node.in_shapes[0]
        params = node.layer.select_param
        layer_attrs = {
            "input_shape": input_shape,
            "point": params.slice_point,
            "axis": params.axis
        }
        self.paddle_graph.add_layer(
            "custom_layer:Select",
            inputs={"x": input.name},
            outputs=layer_outputs,
            **layer_attrs)
