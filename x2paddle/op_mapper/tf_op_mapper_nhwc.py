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

from x2paddle.decoder.tf_decoder import TFGraph
from x2paddle.core.op_mapper import OpMapper
from x2paddle.core.util import *
import math
import inspect
import numpy
import sys


# compute padding size for SAME mode
def get_same_padding(in_size, kernel_size, stride):
    new_size = int(math.ceil(in_size * 1.0 / stride))
    pad_size = (new_size - 1) * stride + kernel_size - in_size
    if pad_size < 0:
        pad_size = 0
    pad0 = int(pad_size / 2)
    pad1 = pad_size - pad0
    return [pad0, pad1]


class TFOpMapperNHWC(OpMapper):
    directly_map_ops = {
        'Relu': ['relu'],
        'Relu6': ['relu6'],
        'Shape': ['shape'],
        'Abs': ['abs'],
        'Sigmoid': ['sigmoid'],
        'Exp': ['exp'],
        'Rsqrt': ['rsqrt'],
        'Sqrt': ['sqrt'],
        'swish_f32': ['swish'],
        'Tanh': ['tanh'],
        'Softplus': ['softplus'],
        'LeakyRelu': ['leaky_relu', {
            'alpha': 'alpha'
        }],
        'Floor': ['floor'],
        'Erf': ['erf']
    }
    elementwise_ops = {
        'Add': 'elementwise_add',
        'AddV2': 'elementwise_add',
        'RealDiv': 'elementwise_div',
        'Sub': 'elementwise_sub',
        'Maximum': 'elementwise_max',
        'Minimum': 'elementwise_min',
        'LessEqual': 'less_equal',
        'Mul': 'elementwise_mul',
        'FloorDiv': 'elementwise_floordiv'
    }

    def __init__(self, decoder):
        super(TFOpMapperNHWC, self).__init__()
        self.decoder = decoder
        self.graph = decoder.tf_graph
        self.weights = dict()
        self.batch_node = None
        self.omit_nodes = list()
        self.used_custom_layers = dict()

        not_placeholder = list()
        for name in self.graph.input_nodes:
            if self.graph.get_node(
                    name).layer_type != "Placeholder" and self.graph.get_node(
                        name
                    ).layer_type != "OneShotIterator" and self.graph.get_node(
                        name).layer_type != "IteratorV2":
                not_placeholder.append(name)
        for name in not_placeholder:
            idx = self.graph.input_nodes.index(name)
            del self.graph.input_nodes[idx]

        unsupported_ops = set()
        sys.stderr.write("Total nodes: {}\n".format(len(self.graph.topo_sort)))
        for i, node_name in enumerate(self.graph.topo_sort):
            sys.stderr.write("\rConverting node {} ...     ".format(i + 1))
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if op in self.directly_map_ops:
                if len(unsupported_ops) > 0:
                    continue
                self.directly_map(node)
            elif op in self.elementwise_ops:
                if len(unsupported_ops) > 0:
                    continue
                self.elementwise_map(node)
            elif hasattr(self, op):
                if len(unsupported_ops) > 0:
                    continue
                func = getattr(self, op)
                try:
                    func(node)
                except Exception as e:
                    unsupported_ops.add(op)
                    print(e)
            else:
                unsupported_ops.add(op)
        if len(unsupported_ops) > 0:
            print("========= {} OPs are not supported yet ===========".format(
                len(unsupported_ops)))
            for op in unsupported_ops:
                print("========== {} ============".format(op))
            sys.exit(-1)
        sys.stderr.write("\nDone!\n")

    def add_omit_nodes(self, in_node_name, out_node_name):
        in_node = self.graph.get_node(in_node_name)
        out_node = self.graph.get_node(out_node_name)
        index = in_node.outputs.index(out_node_name)
        del in_node.outputs[index]
        index = out_node.inputs.index(in_node_name)
        del out_node.inputs[index]
        self.omit_nodes.append(in_node.layer_name)

    def directly_map(self, node):
        assert node.layer_type in self.directly_map_ops
        op_info = self.directly_map_ops[node.layer_type]
        input = self.graph.get_node(node.layer.input[0], copy=True)
        attr = dict()
        for param in op_info[1:]:
            tf_param_name = list(param.keys())[0]
            pd_param_name = list(param.values())[0]
            tf_param = node.get_attr(tf_param_name)
            attr[pd_param_name] = tf_param

        if len(input.out_shapes[0]) == 4 and op_info[0] != 'shape':
            attr1 = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer(
                'transpose', inputs=input, output=node, param_attr=attr1)
            input = node
            node.fluid_code.add_layer(
                op_info[0], inputs=input, output=node, param_attr=attr)
            input = node
            attr2 = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer(
                'transpose', inputs=input, output=node, param_attr=attr2)
        else:
            node.fluid_code.add_layer(
                op_info[0], inputs=input, output=node, param_attr=attr)

    def elementwise_map(self, node):
        assert node.layer_type in self.elementwise_ops
        op_type = self.elementwise_ops[node.layer_type]
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {"x": x, "y": y}
        node.fluid_code.add_layer(
            op_type, inputs=inputs, output=node, param_attr=None)

    def Placeholder(self, node):
        shape = node.out_shapes[0]
        assert len(shape) != 0, "Unknown shape of input nodes[{}].".format(
            node.layer_name)
        dtype = node.dtype
        if shape[0] < 0:
            self.batch_node = node
        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name),
            'append_batch_size': False
        }

        node.fluid_code.add_layer(
            "data", inputs=None, output=node, param_attr=attr)

    def Const(self, node):
        shape = node.out_shapes[0]
        dtype = node.dtype
        value = node.value
        initializer = "Constant(0.0)"
        if len(shape) == 0:
            assert value.size == 1, "Unexpected situation happend"
            shape = [1]
            initializer = "Constant({})".format(value)

        self.weights[node.layer_name] = node.value

        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name),
            'default_initializer': initializer
        }
        node.fluid_code.add_layer(
            "create_parameter", inputs=None, output=node, param_attr=attr)

    def Transpose(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        perm = self.graph.get_node(node.layer.input[1], copy=True)
        assert perm.layer_type == "Const", "Perm of transpose OP should be Const"
        del self.weights[perm.layer_name.replace('/', '_')]
        perm.fluid_code.clear()
        perm = perm.value.tolist()

        attr = {'perm': perm}
        node.fluid_code.add_layer(
            "transpose", inputs=input, output=node, param_attr=attr)

    def Fill(self, node):
        dims = self.graph.get_node(node.layer.input[0], copy=True)
        input_value = self.graph.get_node(node.layer.input[1], copy=True)

        assert input_value.layer_type == "Const", "Value of fill OP should be Const"

        self.add_omit_nodes(input_value.layer_name, node.layer_name)
        input_value = input_value.value
        input_dtype = string(input_value.dtype)
        attr = {'value': input_value, 'dtype': input_dtype}

        node.fluid_code.add_layer(
            "fill_constant", inputs=dims, output=node, param_attr=attr)

    def DepthToSpace(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)

        block_size = node.get_attr("block_size")
        data_format = node.get_attr("data_format").decode()

        if data_format == "NHWC":
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer(
                "transpose", inputs=input, output=input, param_attr=attr)
        n, h, w, c = input.out_shapes[0]

        attr = {'shape': [0, block_size * block_size, -1, h, w]}
        node.fluid_code.add_layer(
            "reshape", inputs=input, output=input, param_attr=attr)

        attr = {'perm': [0, 2, 1, 3, 4]}
        node.fluid_code.add_layer(
            "transpose", inputs=input, output=input, param_attr=attr)
        attr = {'shape': [0, c, h, w]}
        node.fluid_code.add_layer(
            "reshape", inputs=input, output=input, param_attr=attr)

        attr = {'upscale_factor': block_size}
        node.fluid_code.add_layer(
            "pixel_shuffle", inputs=input, output=node, param_attr=attr)

        if data_format == "NHWC":
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer(
                "transpose", inputs=node, output=node, param_attr=attr)

    def MaxPool(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)

        k_size = node.get_attr("ksize")
        strides = node.get_attr("strides")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"

        if not channel_first:
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer(
                "transpose", inputs=input, output=node, param_attr=attr)
            strides = [strides[i] for i in [0, 3, 1, 2]]
            k_size = [k_size[i] for i in [0, 3, 1, 2]]
            input = node

        attr = {
            "pool_size": k_size[2:4],
            "pool_type": string("max"),
            "pool_stride": strides[2:4],
            "pool_padding": string(pad_mode)
        }
        node.fluid_code.add_layer(
            "pool2d", inputs=input, output=node, param_attr=attr)

        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer(
                "transpose", inputs=node, output=node, param_attr=attr)

    def Conv2D(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        kernel = self.graph.get_node(node.layer.input[1], copy=True)
        self.add_omit_nodes(kernel.layer_name, node.layer_name)

        k_size = kernel.out_shapes[0]
        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"

        if kernel.layer_type == 'Const':
            kernel_value = kernel.value
            kernel_weight_name = kernel.layer_name.replace('/', '_')
        else:
            kernel_value = self.decoder.infer_tensor(kernel)
            if kernel.layer_type == 'Split':
                kernel_weight_name = "{}_{}_kernel".format(node.layer_name,
                                                           kernel.layer_name)
            else:
                kernel_weight_name = kernel.layer_name.replace('/', '_')
        self.weights[kernel_weight_name] = numpy.transpose(kernel_value,
                                                           (3, 2, 0, 1))

        if not channel_first:
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer(
                "transpose", inputs=input, output=node, param_attr=attr)
            input = node
        attr = {
            "bias_attr": False,
            "param_attr": string(kernel_weight_name),
            "num_filters": k_size[3],
            "filter_size": k_size[0:2],
            "stride": strides[2:4],
            "dilation": dilations[2:4],
            "padding": string(pad_mode)
        }

        if hasattr(node, 'dilation') and attr['dilation'] == [1, 1]:
            if len(node.dilation) == 1:
                attr['dilation'] = [1, node.dilation[0]]
        node.fluid_code.add_layer(
            "conv2d", inputs=input, output=node, param_attr=attr)
        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer(
                "transpose", inputs=node, output=node, param_attr=attr)

    def BiasAdd(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        bias = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {"x": input, "y": bias}
        node.fluid_code.add_layer(
            "elementwise_add", inputs=inputs, output=node, param_attr=None)

    def FusedBatchNorm(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        gamma = self.graph.get_node(node.layer.input[1], copy=True)
        beta = self.graph.get_node(node.layer.input[2], copy=True)
        moving_mean = self.graph.get_node(node.layer.input[3], copy=True)
        moving_var = self.graph.get_node(node.layer.input[4], copy=True)
        data_format = node.get_attr("data_format").decode()
        channel_first = data_format == "NCHW"

        assert gamma.layer_type == "Const"
        assert beta.layer_type == "Const"
        assert moving_mean.layer_type == "Const"
        assert moving_var.layer_type == "Const"
        self.add_omit_nodes(gamma.layer_name, node.layer_name)
        self.add_omit_nodes(beta.layer_name, node.layer_name)
        self.add_omit_nodes(moving_mean.layer_name, node.layer_name)
        self.add_omit_nodes(moving_var.layer_name, node.layer_name)

        if not channel_first:
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer(
                "transpose", inputs=input, output=node, param_attr=attr)
            input = node

        attr = {
            "epsilon": node.get_attr("epsilon"),
            "param_attr": string(gamma.layer_name),
            "bias_attr": string(beta.layer_name),
            "moving_mean_name": string(moving_mean.layer_name),
            "moving_variance_name": string(moving_var.layer_name),
            "is_test": True
        }

        node.fluid_code.add_layer(
            "batch_norm", inputs=input, output=node, param_attr=attr)

        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer(
                "transpose", inputs=node, output=node, param_attr=attr)

    def DepthwiseConv2dNative(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        kernel = self.graph.get_node(node.layer.input[1], copy=True)
        assert kernel.layer_type == "Const", "Kernel of DepthwiseConv2DNative should be Const"
        self.add_omit_nodes(kernel.layer_name, node.layer_name)

        in_shape = input.out_shapes[0]
        k_size = kernel.out_shapes[0]
        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"

        self.weights[kernel.layer_name.replace('/', '_')] = numpy.transpose(
            kernel.value, (2, 3, 0, 1))

        if not channel_first:
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer(
                "transpose", inputs=input, output=node, param_attr=attr)
            input = node

        attr = {
            "bias_attr": False,
            "param_attr": string(kernel.layer_name),
            "num_filters": in_shape[1],
            "filter_size": k_size[0:2],
            "stride": strides[2:4],
            "dilation": dilations[2:4],
            "groups": k_size[3] * in_shape[1],
            "use_cudnn": False,
            "padding": string(pad_mode)
        }
        node.fluid_code.add_layer(
            "conv2d", inputs=input, output=node, param_attr=attr)

        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer(
                "transpose", inputs=node, output=node, param_attr=attr)

    def Reshape(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        param = self.graph.get_node(node.layer.input[1], copy=True)
        if param.layer_type == "Const":
            self.add_omit_nodes(param.layer_name, node.layer_name)
            shape = param.value.tolist()
        else:
            shape = param
        inputs = {"x": input, "shape": shape}
        node.fluid_code.add_layer(
            "reshape", inputs=inputs, output=node, param_attr=None)
        if param.layer_type != "Const":
            out_shape = numpy.array(node.out_shapes[0])
            if (out_shape > 0).any():
                out_shape[out_shape < 0] = 0
                attr = {'shape': out_shape.tolist()}
                node.fluid_code.add_layer(
                    "reshape", inputs=node, output=node, param_attr=attr)

    def AvgPool(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)

        k_size = node.get_attr("ksize")
        strides = node.get_attr("strides")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"

        if not channel_first:
            strides = [strides[i] for i in [0, 3, 1, 2]]
            k_size = [k_size[i] for i in [0, 3, 1, 2]]
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer(
                "transpose", inputs=input, output=node, param_attr=attr)
            input = node

        attr = {
            "pool_size": k_size[2:4],
            "pool_type": string("avg"),
            "pool_stride": strides[2:4],
            "pool_padding": string(pad_mode)
        }
        node.fluid_code.add_layer(
            "pool2d", inputs=input, output=node, param_attr=attr)

        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer(
                "transpose", inputs=node, output=node, param_attr=attr)

    def SplitV(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        num_sections = self.graph.get_node(node.layer.input[1], copy=True)
        dim = self.graph.get_node(node.layer.input[2], copy=True)
        assert num_sections.layer_type == "Const"
        assert dim.layer_type == "Const"
        self.add_omit_nodes(num_sections.layer_name, node.layer_name)
        self.add_omit_nodes(dim.layer_name, node.layer_name)
        dim = dim.value
        attr = {
            "num_or_sections": num_sections.value.tolist(),
            "dim": dim.value
        }
        node.fluid_code.add_layer(
            "split", inputs=input, output=node, param_attr=attr)

    def ConcatV2(self, node):
        inputs = [
            self.graph.get_node(
                name, copy=True) for name in node.layer.input[:-1]
        ]
        axis = self.graph.get_node(node.layer.input[-1], copy=True)
        assert axis.layer_type == "Const"
        self.add_omit_nodes(axis.layer_name, node.layer_name)
        axis = axis.value
        if axis < 0:
            axis += len(inputs[0].out_shapes[0])
        attr = {"axis": axis}
        node.fluid_code.add_layer(
            "concat", inputs=inputs, output=node, param_attr=attr)

    def Tile(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        expand_times = self.graph.get_node(node.layer.input[1], copy=True)
        if expand_times.layer_type == "Const":
            self.add_omit_nodes(expand_times.layer_name, node.layer_name)
            expand_times = expand_times.value.tolist()
        else:
            expand_times = expand_times
        inputs = {"x": input, "expand_times": expand_times}
        node.fluid_code.add_layer(
            "expand", inputs=inputs, output=node, param_attr=None)

    def Pack(self, node):
        inputs = [
            self.graph.get_node(
                name, copy=True) for name in node.layer.input
        ]
        reshape_shape = list()
        for input_node in inputs:
            k_size = input_node.out_shapes[0]
            if len(k_size) and k_size[-1] != -1:
                reshape_shape = [0] * len(k_size)
                reshape_shape[-1] = k_size[-1]
                break
        if len(reshape_shape):
            for i, input_node in enumerate(inputs):
                node.fluid_code.add_layer(
                    "reshape",
                    inputs=input_node,
                    output='tmp_{}'.format(i),
                    param_attr={"shape": reshape_shape})
        axis = node.get_attr("axis")
        attr = {"axis": axis}
        if len(reshape_shape):
            inputs = ['tmp_{}'.format(i) for i in range(len(inputs))]
        node.fluid_code.add_layer(
            "stack", inputs=inputs, output=node, param_attr=attr)

    def Pad(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        paddings = self.graph.get_node(node.layer.input[1], copy=True)
        assert paddings.layer_type == "Const", "Padding should be Const"
        self.add_omit_nodes(paddings.layer_name, node.layer_name)
        paddings = paddings.value.flatten().tolist()
        data_format = input.tf_data_format

        if len(input.out_shapes[0]) == 4:
            new_padding = None
            if input.tf_data_format == "NHWC":
                if paddings[0] + paddings[1] + paddings[6] + paddings[7] == 0:
                    new_padding = paddings[2:6]
            else:
                if paddings[0] + paddings[1] + paddings[2] + paddings[3] == 0:
                    new_padding = paddings[4:]
            if new_padding is not None:
                if input.tf_data_format == "NHWC":
                    attr = {"perm": [0, 3, 1, 2]}
                    node.fluid_code.add_layer(
                        "transpose", inputs=input, output=node, param_attr=attr)
                    input = node
                attr = {"paddings": new_padding}
                node.fluid_code.add_layer(
                    "pad2d", inputs=input, output=node, param_attr=attr)
                if input.tf_data_format == "NHWC":
                    attr = {"perm": [0, 2, 3, 1]}
                    node.fluid_code.add_layer(
                        "transpose", inputs=node, output=node, param_attr=attr)

                return

        attr = {"paddings": paddings}
        node.fluid_code.add_layer(
            "pad", inputs=input, output=node, param_attr=attr)

    def Range(self, node):
        start = self.graph.get_node(node.layer.input[0], copy=True)
        limit = self.graph.get_node(node.layer.input[1], copy=True)
        delta = self.graph.get_node(node.layer.input[2], copy=True)

        if start.layer_type == "Const":
            self.add_omit_nodes(start.layer_name, node.layer_name)
            start = start.value
        if limit.layer_type == "Const":
            self.add_omit_nodes(limit.layer_name, node.layer_name)
            limit = limit.value
        if delta.layer_type == "Const":
            self.add_omit_nodes(delta.layer_name, node.layer_name)
            delta = delta.value

        dtype = node.dtype
        inputs = {
            "start": start,
            "end": limit,
            "step": delta,
        }
        attr = {"dtype": string(node.dtype)}
        node.fluid_code.add_layer(
            "range", inputs=inputs, output=node, param_attr=attr)

    def Mean(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        reduce_idx = self.graph.get_node(node.layer.input[1], copy=True)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        dims = reduce_idx.value.tolist()
        keep_dims = node.get_attr("keep_dims")

        attr = {"dim": dims, "keep_dim": keep_dims}
        node.fluid_code.add_layer(
            "reduce_mean", inputs=input, output=node, param_attr=attr)

    def MatMul(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        transpose_a = node.get_attr('transpose_a')
        transpose_b = node.get_attr('transpose_b')
        inputs = {"x": x, "y": y}
        # fix paddle shape infer problem
        # should be removed after paddle 1.6
        if x.out_shapes[0][-1] < 0 and y.out_shapes[0][0] > 0:
            shape = x.out_shapes[0]
            shape[-1] = y.out_shapes[0][0]
            attr = {"shape": shape}
            node.fluid_code.add_layer(
                "reshape", inputs=x, output=x, param_attr=attr)
        if transpose_a is None:
            transpose_a = node.get_attr('adj_x')
        if transpose_b is None:
            transpose_b = node.get_attr('adj_y')
        attr = {"transpose_x": transpose_a, "transpose_y": transpose_b}
        node.fluid_code.add_layer(
            "matmul", inputs=inputs, output=node, param_attr=attr)

    def BatchMatMul(self, node):
        return self.MatMul(node)

    def BatchMatMulV2(self, node):
        return self.MatMul(node)

    def ArgMax(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        axis = self.graph.get_node(node.layer.input[1], copy=True)
        assert axis.layer_type == "Const", "ArgMax only support Const parameter"
        self.add_omit_nodes(axis.layer_name, node.layer_name)
        axis = axis.value
        attr = {"axis": axis}
        node.fluid_code.add_layer(
            "argmax", inputs=input, output=node, param_attr=attr)

    def StridedSlice(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        begin = self.graph.get_node(node.layer.input[1], copy=True)
        end = self.graph.get_node(node.layer.input[2], copy=True)
        strides = self.graph.get_node(node.layer.input[3], copy=True)
        assert begin.layer_type == "Const"
        assert end.layer_type == "Const"
        assert strides.layer_type == "Const"
        self.add_omit_nodes(begin.layer_name, node.layer_name)
        self.add_omit_nodes(end.layer_name, node.layer_name)
        self.add_omit_nodes(strides.layer_name, node.layer_name)
        strides = strides.value.tolist()
        assert len(set(strides)) == 1 and strides[
            0] == 1, "Only support strides be 1 in StridedSlice OP"

        begin = begin.value.tolist()
        end = end.value.tolist()

        for i in range(len(end)):
            if end[i] == 0:
                end[i] = 999999

        begin_mask = node.get_attr('begin_mask')
        end_mask = node.get_attr('end_mask')
        ellipsis_mask = node.get_attr('ellipsis_mask')
        new_axis_mask = node.get_attr('new_axis_mask')
        shrink_axis_mask = node.get_attr('shrink_axis_mask')

        assert ellipsis_mask == 0, "(OP:{} Name:{})Only support ellipsis_mask be 0[now: {}] n StridedSlice OP".format(
            node.layer_type, node.layer.name, ellipsis_mask)

        # TODO codes without validation
        # Use it carefully
        new_begin = list()
        new_end = list()
        new_axes = list()
        shrink_axes = list()
        for i, item in enumerate(begin):
            mask = (new_axis_mask >> i) & 1
            if mask != 0:
                new_axes.append(i)
                continue

            mask = (shrink_axis_mask >> i) & 1
            if mask != 0:
                shrink_axes.append(i)

            mask = (begin_mask >> i) & 1
            if mask != 0:
                new_begin.append(0)
            else:
                new_begin.append(item)

            mask = (end_mask >> i) & 1
            if mask != 0:
                new_end.append(999999)
            else:
                new_end.append(end[i])

        attr = {
            "axes": [i for i in range(len(new_begin))],
            "starts": new_begin,
            "ends": new_end
        }
        node.fluid_code.add_layer(
            "slice", inputs=input, output=node, param_attr=attr)
        if len(new_axes) > 0:
            attr = {"axes": new_axes}
            node.fluid_code.add_layer(
                "unsqueeze", inputs=node, output=node, param_attr=attr)
        if len(shrink_axes) > 0:
            if len(input.out_shapes[0]) + len(new_axes) <= 1:
                pass
            else:
                attr = {"axes": shrink_axes}
                node.fluid_code.add_layer(
                    "squeeze", inputs=node, output=node, param_attr=attr)

    def Slice(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        begin = self.graph.get_node(node.layer.input[1], copy=True)
        size = self.graph.get_node(node.layer.input[2], copy=True)
        if begin.layer_type == "Const":
            self.add_omit_nodes(begin.layer_name, node.layer_name)
            begin = begin.value.tolist()
        else:
            begin = begin
            shape = begin.out_shapes[0]
            attr = {"shape": shape}
            node.fluid_code.add_layer(
                "reshape", inputs=begin, output=begin, param_attr=attr)
        if size.layer_type == "Const":
            self.add_omit_nodes(size.layer_name, node.layer_name)
            size = size.value.tolist()
        else:
            size = size
            shape = size.out_shapes[0]
            attr = {"shape": shape}
            node.fluid_code.add_layer(
                "reshape", inputs=size, output=size, param_attr=attr)
        inputs = {"x": input, "offsets": begin, "shape": size}
        node.fluid_code.add_layer(
            "crop_tensor", inputs=inputs, output=node, param_attr=None)

    def Conv2DBackpropInput(self, node):
        out_shape = self.graph.get_node(node.layer.input[0], copy=True)
        kernel = self.graph.get_node(node.layer.input[1], copy=True)
        input = self.graph.get_node(node.layer.input[2], copy=True)

        assert kernel.layer_type == "Const", "Kernel of Conv2DBackpropInput should be Const"

        self.add_omit_nodes(kernel.layer_name, node.layer_name)
        self.add_omit_nodes(out_shape.layer_name, node.layer_name)

        if out_shape.layer_type == "Const":
            out_shape = out_shape.value.tolist()
        else:
            out_shape = self.decoder.infer_shape_tensor(out_shape,
                                                        node.out_shapes[0])

        in_shape = input.out_shapes[0]
        if in_shape.count(-1) > 2:
            in_shape = self.decoder.infer_tensor(input).shape
        k_size = kernel.out_shapes[0]
        if k_size.count(-1) > 2:
            k_size = self.decoder.infer_tensor(kernel).shape

        pad_mode = node.get_attr("padding").decode()
        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        channel_first = data_format == "NCHW"

        self.weights[kernel.layer_name.replace('/', '_')] = numpy.transpose(
            kernel.value, (3, 2, 0, 1))
        if not channel_first:
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer(
                "transpose", inputs=input, output=node, param_attr=attr)
            input = node
        else:
            self.graph.data_format_propagation(node)

        attr = {
            "bias_attr": False,
            "param_attr": string(kernel.layer_name),
            "num_filters": k_size[2],
            "filter_size": k_size[0:2],
            "stride": strides[2:4],
            "dilation": dilations[2:4],
            "padding": string(pad_mode),
            "output_size": out_shape[1:3]
        }
        node.fluid_code.add_layer(
            "conv2d_transpose", inputs=input, output=node, param_attr=attr)

        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer(
                "transpose", inputs=node, output=node, param_attr=attr)

    def Max(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        reduce_idx = self.graph.get_node(node.layer.input[1], copy=True)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        keep_dims = node.get_attr("keep_dims")
        dim = reduce_idx.value.tolist()

        attr = {"dim": dim, "keep_dim": keep_dims}
        node.fluid_code.add_layer(
            "reduce_max", inputs=input, output=node, param_attr=attr)

    def Sum(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        reduce_idx = self.graph.get_node(node.layer.input[1], copy=True)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        keep_dims = node.get_attr("keep_dims")
        dim = reduce_idx.value.tolist()

        attr = {"dim": dim, "keep_dim": keep_dims}
        node.fluid_code.add_layer(
            "reduce_sum", inputs=input, output=node, param_attr=attr)

    def Cast(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        dtype = node.dtype_map[node.get_attr('DstT')]
        attr = {"dtype": string(dtype)}
        node.fluid_code.add_layer(
            "cast", inputs=input, output=node, param_attr=attr)

    def Split(self, node):
        dim = self.graph.get_node(node.layer.input[0], copy=True)
        input = self.graph.get_node(node.layer.input[1], copy=True)
        assert dim.layer_type == "Const"
        self.add_omit_nodes(dim.layer_name, node.layer_name)
        num_split = node.get_attr('num_split')
        dim = dim.value

        attr = {"num_or_sections": num_split, "dim": dim}
        node.fluid_code.add_layer(
            "split", inputs=input, output=node, param_attr=attr)

    def Squeeze(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        squeeze_dims = node.get_attr('squeeze_dims')
        attr = {"axes": squeeze_dims}
        node.fluid_code.add_layer(
            "squeeze", inputs=input, output=node, param_attr=attr)

    def Softmax(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        axis = node.get_attr("axis")
        attr = {"axis": axis}
        node.fluid_code.add_layer(
            "softmax", inputs=input, output=node, param_attr=attr)

    def ResizeNearestNeighbor(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        resize_shape = self.graph.get_node(node.layer.input[1], copy=True)
        if resize_shape.layer_type == "Const":
            self.add_omit_nodes(resize_shape.layer_name, node.layer_name)
            resize_shape = resize_shape.value.tolist()
        else:
            resize_shape = resize_shape
            shape = resize_shape.out_shapes[0]
            attr = {"shape": shape}
            node.fluid_code.add_layer(
                "reshape",
                inputs=resize_shape,
                output=resize_shape,
                param_attr=attr)

        align_corners = node.get_attr("align_corners")
        attr = {"perm": [0, 3, 1, 2]}
        node.fluid_code.add_layer(
            "transpose", inputs=input, output=node, param_attr=attr)
        inputs = {"input": node, "out_shape": resize_shape}
        attr = {"align_corners": align_corners}
        node.fluid_code.add_layer(
            "resize_nearest", inputs=inputs, output=node, param_attr=attr)
        attr = {"perm": [0, 2, 3, 1]}
        node.fluid_code.add_layer(
            "transpose", inputs=node, output=node, param_attr=attr)

    def ResizeBilinear(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        resize_shape = self.graph.get_node(node.layer.input[1], copy=True)
        if resize_shape.layer_type == "Const":
            self.add_omit_nodes(resize_shape.layer_name, node.layer_name)
            resize_shape = resize_shape.value.tolist()
        else:
            shape = resize_shape.out_shapes[0]
            attr = {"shape": shape}
            node.fluid_code.add_layer(
                "reshape",
                inputs=resize_shape,
                output=resize_shape,
                param_attr=attr)
        align_corners = node.get_attr("align_corners")
        attr = {"perm": [0, 3, 1, 2]}
        node.fluid_code.add_layer(
            "transpose", inputs=input, output=node, param_attr=attr)
        inputs = {"input": node, "out_shape": resize_shape}
        attr = {
            #"out_shape": resize_shape,
            "align_corners": align_corners,
            "align_mode": 1
        }
        node.fluid_code.add_layer(
            "resize_bilinear", inputs=inputs, output=node, param_attr=attr)
        attr = {"perm": [0, 2, 3, 1]}
        node.fluid_code.add_layer(
            "transpose", inputs=node, output=node, param_attr=attr)

    def GreaterEqual(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {"x": x, "y": y}
        node.fluid_code.add_layer(
            "greater_equal", inputs=inputs, output=node, param_attr=None)

    def RandomUniform(self, node):
        shape = self.graph.get_node(node.layer.input[0], copy=True)
        if shape.layer_type == "Const":
            self.add_omit_nodes(shape.layer_name, node.layer_name)
            shape = shape.value.tolist()
        else:
            shape = shape
        attr = {"min": 0.0, "max": 0.9999}

        node.fluid_code.add_layer(
            "uniform_random", inputs=shape, output=node, param_attr=attr)

    def SquaredDifference(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {"x": x, "y": y}
        node.fluid_code.add_layer(
            "elementwise_sub", inputs=inputs, output=node, param_attr=None)
        inputs = {"x": node, "y": node}
        node.fluid_code.add_layer(
            "elementwise_mul", inputs=inputs, output=node, param_attr=None)

    def ExpandDims(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        if y.layer_type == 'Const':
            self.add_omit_nodes(y.layer_name, node.layer_name)
            dim = y.value.tolist()
            if not isinstance(dim, list):
                dim = [dim]
            attr = {'axes': dim}
        else:
            attr = {'axes': y}
        node.fluid_code.add_layer(
            "unsqueeze", inputs=x, output=node, param_attr=attr)

    def BatchToSpaceND(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        if hasattr(node, 'skip') and node.skip:
            node.fluid_code.add_layer(
                "=", inputs=x, output=node, param_attr=None)
        else:
            raise Exception("BatchToSpaceND is not supported")

    def SpaceToBatchND(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        if hasattr(node, 'skip') and node.skip:
            node.fluid_code.add_layer(
                "=", inputs=x, output=node, param_attr=None)
        else:
            raise Exception("SpaceToBatchND is not supported")

    def OneHot(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        depth = self.graph.get_node(node.layer.input[1], copy=True)
        on_value = self.graph.get_node(node.layer.input[2], copy=True)
        off_value = self.graph.get_node(node.layer.input[3], copy=True)
        assert depth.layer_type == 'Const', 'Parameter depth should be Const in OneHot'
        assert on_value.layer_type == 'Const', 'Parameter on_value should be Const in OneHot'
        assert off_value.layer_type == 'Const', 'Parameter off_value should be Const in OneHot'
        self.add_omit_nodes(depth.layer_name, node.layer_name)
        self.add_omit_nodes(on_value.layer_name, node.layer_name)
        self.add_omit_nodes(off_value.layer_name, node.layer_name)
        depth = depth.value
        on_value = on_value.value
        off_value = off_value.value
        assert math.fabs(on_value -
                         1.0) < 1e-06, "on_value should be 1 in OneHot"
        assert math.fabs(off_value -
                         0.0) < 1e-06, "off_value should be 0 in OneHot"
        attr = {'depth': depth}
        node.fluid_code.add_layer(
            "one_hot",
            inputs=input,
            output=node,
            param_attr=attr,
            use_fluid=True)

    def Pow(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        factor = self.graph.get_node(node.layer.input[1], copy=True)
        self.add_omit_nodes(factor.layer_name, node.layer_name)
        if factor.layer_type == 'Const':
            factor = factor.value.tolist()
        else:
            factor = self.decoder.infer_tensor(factor)
        attr = {'factor': factor}
        node.fluid_code.add_layer("pow", inputs=x, output=node, param_attr=attr)

    def All(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        reduce_idx = self.graph.get_node(node.layer.input[1], copy=True)
        self.add_omit_nodes(reduce_idx.layer_name, node.layer_name)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        dims = reduce_idx.value.tolist()
        keep_dims = node.get_attr("keep_dims")

        attr = {"dim": dims, "keep_dim": keep_dims}
        node.fluid_code.add_layer(
            "reduce_all", inputs=input, output=node, param_attr=attr)

    def GatherV2(self, node):
        embeddings = self.graph.get_node(node.layer.input[0], copy=True)
        index = self.graph.get_node(node.layer.input[1], copy=True)
        axis = self.graph.get_node(node.layer.input[2], copy=True)
        self.add_omit_nodes(axis.layer_name, node.layer_name)
        assert axis.layer_type == 'Const', "Only support Const parameter[axis]"
        axis = axis.value.tolist()
        assert axis == 0, "Only support axis=0 in GatherV2 OP"
        attr = {'overwrite': False}
        if len(index.out_shapes[0]) != 1:
            reshape_attr = {"shape": [-1]}
            node.fluid_code.add_layer(
                "reshape", inputs=index, output=index, param_attr=reshape_attr)
        inputs = {'input': embeddings, 'index': index}
        node.fluid_code.add_layer(
            "gather", inputs=inputs, output=node, param_attr=attr)

    def OneShotIterator(self, node):
        return self.Placeholder(node)

    def IteratorV2(self, node):
        dtype_map = {
            1: "float32",
            3: "int32",
            4: "uint8",
            9: "int64",
            10: "bool"
        }
        shapes = node.out_shapes
        dtypes = node.layer.attr['output_types'].list.type
        node.fluid_code.add_note("{} = [0] * {}".format(node.layer_name,
                                                        len(shapes)))
        for i, shape in enumerate(shapes):
            attr = {
                'dtype': string(dtype_map[dtypes[i]]),
                'shape': shape,
                'name': string("{}_{}".format(node.layer_name, i)),
                'append_batch_size': False
            }
            output = "{}[{}]".format(node.layer_name, i)
            node.fluid_code.add_layer(
                "data", inputs=None, output=output, param_attr=attr)
