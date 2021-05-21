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

from x2paddle.decoder.tf_decoder import TFGraph, TFGraphNode
from x2paddle.core.program import PaddleGraph
from x2paddle.core.util import *
import traceback
import math
import inspect
import numpy
import sys

name_counter = dict()


def gen_name(op_name, var_name):
    name = "{}_{}".format(op_name, var_name)
    if name not in name_counter:
        name_counter[name] = 0
    else:
        name_counter[name] += 1
    name = name + '_' + str(name_counter[name])
    return name


# compute padding size for SAME mode
def get_same_padding(in_size, kernel_size, stride):
    new_size = int(math.ceil(in_size * 1.0 / stride))
    pad_size = (new_size - 1) * stride + kernel_size - in_size
    if pad_size < 0:
        pad_size = 0
    pad0 = int(pad_size / 2)
    pad1 = pad_size - pad0
    return [pad0, pad1]


class TFOpMapper():
    directly_map_ops = {
        'Relu': ['paddle.nn.ReLU'],
        'Relu6': ['paddle.nn.ReLU6'],
        'Abs': ['paddle.abs'],
        'Sigmoid': ['paddle.nn.Sigmoid'],
        'Exp': ['paddle.exp'],
        'Rsqrt': ['paddle.rsqrt'],
        'Sqrt': ['paddle.sqrt'],
        'swish_f32': ['paddle.nn.Swish'],
        'Tanh': ['paddle.nn.Tanh'],
        'Softplus': ['paddle.nn.Softplus'],
        'LeakyRelu': ['paddle.nn.LeakyReLU', dict(alpha='negative_slope')],
        'Softmax': ['paddle.nn.Softmax'],
        'Floor': ['paddle.floor'],
        'Erf': ['paddle.erf'],
        'Square': ['paddle.square']
    }
    elementwise_ops = {
        'Add': 'paddle.add',
        'AddV2': 'paddle.add',
        'RealDiv': 'paddle.divide',
        'DivNoNan': 'paddle.divide',
        'Sub': 'paddle.subtract',
        'Maximum': 'paddle.maximum',
        'Minimum': 'paddle.minimum',
        'Mul': 'paddle.multiply',
        'FloorDiv': 'paddle.floor_divide',
        'FloorMod': 'paddle.floor_mod',
        'LogicalAnd': 'logical_and',
    }
    bool_ops = {
        'LessEqual': 'paddle.less_equal',
        'GreaterEqual': 'paddle.greater_equal',
        'Greater': 'paddle.greater_than',
        'NotEqual': 'paddle.not_equal',
        'Equal': 'paddle.equal',
    }

    def __init__(self, decoder):
        self.decoder = decoder
        self.graph = decoder.tf_graph
        if not self.op_checker():
            raise Exception("Model is not supported yet.")
        self.params = dict()
        self.nn_name2id = dict()
        self.input_index = 0
        self.inputs_info = dict()
        self.paddle_graph = PaddleGraph(parent_layer=None, source_type="tf")
        self.paddle_graph.outputs = self.graph.output_nodes

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

        print("Total nodes: {}".format(
            sum([
                isinstance(node, TFGraphNode)
                for name, node in self.graph.node_map.items()
            ])))
        print("Nodes converting ...")
        for i, node_name in enumerate(self.graph.topo_sort):
            sys.stderr.write("\rConverting node {} ...     ".format(i + 1))
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if op in self.directly_map_ops:
                self.directly_map(node)
            elif op in self.elementwise_ops:
                self.elementwise_map(node)
            elif op in self.bool_ops:
                self.bool_map(node)
            elif hasattr(self, op):
                func = getattr(self, op)
                func(node)
        print("\nNodes converted.")
        self.paddle_graph.set_name(self.graph.graph_name)
        self.paddle_graph.set_parameters(self.params)
        self.paddle_graph.set_inputs_info(self.inputs_info)

    def op_checker(self):
        unsupported_ops = set()
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if not hasattr(self, op) and \
                op not in self.directly_map_ops and \
                op not in self.elementwise_ops and \
                op not in self.bool_ops:
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
        layer_attrs = dict()
        if len(op_info) > 1:
            attrs_name_map_dict = op_info[1]
            for tf_attr_name, pd_attr_name in attrs_name_map_dict.items():
                layer_attrs[pd_attr_name] = node.get_attr(tf_attr_name)
        if paddle_op.startswith("paddle.nn"):
            op_name = paddle_op[10:].lower()
            op_name = name_generator(op_name, self.nn_name2id)
            output_name = node.name
            layer_outputs = [op_name, output_name]
            self.paddle_graph.add_layer(
                kernel=paddle_op,
                inputs={"x": input.name},
                outputs=layer_outputs,
                **layer_attrs)
        else:
            self.paddle_graph.add_layer(
                kernel=paddle_op,
                inputs={"x": input.name},
                outputs=[node.name],
                **layer_attrs)

    def elementwise_map(self, node, op_type=None):
        if op_type is None:
            assert node.layer_type in self.elementwise_ops
            op_type = self.elementwise_ops[node.layer_type]
        x = self.graph.get_input_node(node, 0)
        y = self.graph.get_input_node(node, 1)
        x_shape = x.out_shapes[0]
        y_shape = y.out_shapes[0]
        layer_id = self.paddle_graph.add_layer(
            kernel=op_type,
            inputs={"x": x.name,
                    "y": y.name},
            outputs=[node.name])
        self.paddle_graph.layers[layer_id].input_shapes = {
            "x": x_shape,
            "y": y_shape
        }

    def bool_map(self, node):
        op_type = self.bool_ops[node.layer_type]
        self.elementwise_map(node, op_type)
        node.set_dtype("bool")

    def Placeholder(self, node):
        shape = node.out_shapes[0]
        assert len(shape) != 0, "Unknown shape of input nodes[{}].".format(
            node.layer_name)
        dtype = node.dtype

        self.paddle_graph.add_layer(
            kernel="paddle.to_tensor",
            inputs={},
            outputs=[node.name],
            data="x{}".format(self.input_index))
        self.inputs_info["x{}".format(self.input_index)] = [shape, node.dtype]
        self.input_index += 1

    def Const(self, node):
        shape = node.out_shapes[0]
        dtype = node.dtype
        value = node.value
        if len(shape) == 0:
            assert value.size == 1, "Unexpected situation happend"
            if value == float('inf'):
                value = "float('inf')"
            self.paddle_graph.add_layer(
                "paddle.full",
                inputs={},
                outputs=[node.name],
                dtype=string(dtype),
                shape=[1],
                fill_value=value)
            return
        self.params[node.name] = node.value

        if 0 not in shape:
            self.paddle_graph.add_layer(
                "self.create_parameter",
                inputs={},
                outputs=[node.name],
                shape=shape,
                attr=string(node.name),
                dtype=string(dtype),
                default_initializer="paddle.nn.initializer.Constant(value=0.0)")

    def Transpose(self, node):
        input = self.graph.get_input_node(node, 0)
        perm = self.graph.get_input_node(node, 1)
        if perm.layer_type == "Const":
            perm = perm.value.tolist()
        else:
            perm = self.decoder.infer_tensor(
                perm, use_diff_inputs=False).tolist()

        self.paddle_graph.add_layer(
            "paddle.transpose",
            inputs={"x": input.name},
            outputs=[node.name],
            perm=perm)

    def Where(self, node):
        if len(node.layer.input) == 1:
            cond = self.graph.get_input_node(node, 0)
            self.paddle_graph.add_layer(
                "paddle.nonzero", inputs={"x": cond.name}, outputs=[node.name])
        else:
            cond = self.graph.get_input_node(node, 0)
            x = self.graph.get_input_node(node, 1)
            y = self.graph.get_input_node(node, 2)
            self.paddle_graph.add_layer(
                "paddle.where",
                inputs={"condition": cond.name,
                        "x": x.name,
                        "y": y.name},
                outputs=[node.name])

    def Neg(self, node):
        input = self.graph.get_input_node(node, 0)

        self.paddle_graph.add_layer(
            "paddle.scale",
            inputs={"x": input.name},
            outputs=[node.name],
            scale=-1)

    def Fill(self, node):
        dims = self.graph.get_input_node(node, 0)
        input_value = self.graph.get_input_node(node, 1)
        inputs = dict()
        layer_attrs = dict()
        assert input_value.layer_type == "Const", "Value of fill OP should be Const"
        if dims.layer_type == "Const":
            layer_attrs["shape"] = dims.value.tolist()
        else:
            inputs["shape"] = dims.name
        layer_attrs["dtype"] = string(input_value.dtype)
        layer_attrs["fill_value"] = input_value.value

        self.paddle_graph.add_layer(
            "paddle.full", inputs=inputs, outputs=[node.name], **layer_attrs)

    def DepthToSpace(self, node):
        input = self.graph.get_input_node(node, 0)

        block_size = node.get_attr("block_size")
        data_format = node.get_attr("data_format").decode()
        if data_format == "NHWC":
            n, h, w, c = input.out_shapes[0]
        else:
            n, c, h, w = input.out_shapes[0]

        input_name = input.name
        if data_format == "NHWC":
            transpose_name = gen_name("depth_to_space", "transpose")
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": input.name},
                outputs=[transpose_name],
                perm=[0, 3, 1, 2])
            input_name = transpose_name

        shape = [0, block_size * block_size, -1, h, w]
        reshape_name = gen_name("depth_to_space", "reshape")
        self.paddle_graph.add_layer(
            kernel="paddle.reshape",
            inputs={"x": input_name},
            outputs=[reshape_name],
            shape=shape)

        transpose_name = gen_name("depth_to_space", "transpose")
        self.paddle_graph.add_layer(
            kernel="paddle.transpose",
            inputs={"x": reshape_name},
            outputs=[transpose_name],
            perm=[0, 2, 1, 3, 4])

        reshape_name = gen_name("depth_to_space", "reshape")
        self.paddle_graph.add_layer(
            kernel="paddle.reshape",
            inputs={"x": transpose_name},
            outputs=[reshape_name],
            shape=[0, c, h, w])

        self.paddle_graph.add_layer(
            kernel="paddle.nn.functional.pixel_shuffle",
            inputs={"x": reshape_name},
            outputs=[node.name],
            upscale_factor=block_size)

        if data_format == "NHWC":
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": node.name},
                outputs=[node.name],
                perm=[0, 2, 3, 1])

    def MaxPool(self, node):
        input = self.graph.get_input_node(node, 0)

        k_size = node.get_attr("ksize")
        strides = node.get_attr("strides")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()

        input_name = input.name
        if data_format == "NHWC":
            transpose_name = gen_name("max_pool", "transpose")
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": input.name},
                outputs=[transpose_name],
                perm=[0, 3, 1, 2])
            strides = [strides[i] for i in [0, 3, 1, 2]]
            k_size = [k_size[i] for i in [0, 3, 1, 2]]
            input_name = transpose_name

        op_name = name_generator("pool", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]

        self.paddle_graph.add_layer(
            kernel="paddle.nn.MaxPool2D",
            inputs={"input": input_name},
            outputs=layer_outputs,
            kernel_size=k_size[2:4],
            stride=strides[2:4],
            padding=string(pad_mode))

        if data_format == "NHWC":
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": node.name},
                outputs=[node.name],
                perm=[0, 2, 3, 1])

    def Conv2D(self, node):
        op_name = name_generator("conv", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        input = self.graph.get_input_node(node, 0)
        kernel = self.graph.get_input_node(node, 1)

        k_size = kernel.out_shapes[0]
        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        if data_format == "NHWC":
            n, h, w, c = input.out_shapes[0]
        else:
            n, c, h, w = input.out_shapes[0]

        if kernel.layer_type == 'Const':
            kernel_value = kernel.value
        else:
            kernel_value = self.decoder.infer_tensor(
                kernel, use_diff_inputs=False)
        kernel_weight_name = op_name + ".weight"
        self.params[kernel_weight_name] = numpy.transpose(kernel_value,
                                                          (3, 2, 0, 1))

        input_name = input.name
        if data_format == "NHWC":
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]
            transpose_name = gen_name("conv2d", "transpose")
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": input.name},
                outputs=[transpose_name],
                perm=[0, 3, 1, 2])
            input_name = transpose_name

        if c == -1:
            attr = {"shape": [0, k_size[2], 0, 0]}
            self.paddle_graph.add_layer(
                kernel="paddle.reshape",
                inputs={"x": input_name},
                outputs=[input_name],
                shape=[0, k_size[2], 0, 0])

        self.paddle_graph.add_layer(
            kernel="paddle.nn.Conv2D",
            inputs={"input": input_name},
            outputs=layer_outputs,
            weight_attr=string(kernel_weight_name),
            bias_attr=False,
            in_channels=k_size[2],
            out_channels=k_size[3],
            kernel_size=k_size[0:2],
            stride=strides[2:4],
            dilation=dilations[2:4],
            padding=string(pad_mode))

        if data_format == "NHWC":
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": node.name},
                outputs=[node.name],
                perm=[0, 2, 3, 1])

    def Conv3D(self, node):
        op_name = name_generator("conv", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        input = self.graph.get_input_node(node, 0)
        kernel = self.graph.get_input_node(node, 1)

        k_size = kernel.out_shapes[0]
        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        if data_format == "NDHWC":
            n, d, h, w, c = input.out_shapes[0]
        else:
            n, c, d, h, w = input.out_shapes[0]

        if kernel.layer_type == 'Const':
            kernel_value = kernel.value
        else:
            kernel_value = self.decoder.infer_tensor(
                kernel, use_diff_inputs=False)
        kernel_weight_name = op_name + ".weight"
        self.params[kernel_weight_name] = numpy.transpose(kernel_value,
                                                          (4, 3, 0, 1, 2))

        input_name = input.name
        if data_format == "NDHWC":
            strides = [strides[i] for i in [0, 4, 1, 2, 3]]
            dilations = [dilations[i] for i in [0, 4, 1, 2, 3]]
            transpose_name = gen_name("conv3d", "transpose")
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": input.name},
                outputs=[transpose_name],
                perm=[0, 4, 1, 2, 3])
            input_name = transpose_name

        if c == -1:
            attr = {"shape": [0, k_size[2], 0, 0, 0]}
            self.paddle_graph.add_layer(
                kernel="paddle.reshape",
                inputs={"x": input_name},
                outputs=[input_name],
                shape=[0, k_size[2], 0, 0, 0])

        self.paddle_graph.add_layer(
            kernel="paddle.nn.Conv3D",
            inputs={"input": input_name},
            outputs=layer_outputs,
            weight_attr=string(kernel_weight_name),
            bias_attr=False,
            in_channels=k_size[3],
            out_channels=k_size[4],
            kernel_size=k_size[0:3],
            stride=strides[2:5],
            dilation=dilations[2:5],
            padding=string(pad_mode))

        if data_format == "NDHWC":
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": node.name},
                outputs=[node.name],
                perm=[0, 2, 3, 4, 1])

    def BiasAdd(self, node):
        input = self.graph.get_input_node(node, 0)
        bias = self.graph.get_input_node(node, 1)
        self.paddle_graph.add_layer(
            kernel="paddle.add",
            inputs={"x": input.name,
                    "y": bias.name},
            outputs=[node.name])

    def FusedBatchNorm(self, node):
        op_name = name_generator("bn", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        input = self.graph.get_input_node(node, 0)

        gamma = self.graph.get_input_node(node, 1)
        beta = self.graph.get_input_node(node, 2)
        moving_mean = self.graph.get_input_node(node, 3)
        moving_var = self.graph.get_input_node(node, 4)
        data_format = node.get_attr("data_format").decode()

        assert gamma.layer_type == "Const"
        assert beta.layer_type == "Const"
        assert moving_mean.layer_type == "Const"
        assert moving_var.layer_type == "Const"

        input_name = input.name
        if data_format == "NHWC":
            transpose_name = gen_name("batch_norm", "transpose")
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": input.name},
                outputs=[transpose_name],
                perm=[0, 3, 1, 2])
            input_name = transpose_name
            n, h, w, c = input.out_shapes[0]
        else:
            n, c, h, w = input.out_shapes[0]

        self.params["{}_{}".format(node.name, gamma.name)] = self.params[
            gamma.name]
        self.params["{}_{}".format(node.name, beta.name)] = self.params[
            beta.name]
        self.params["{}_{}".format(node.name, moving_mean.name)] = self.params[
            moving_mean.name]
        self.params["{}_{}".format(node.name, moving_var.name)] = self.params[
            moving_var.name]
        self.paddle_graph.add_layer(
            kernel="paddle.nn.BatchNorm",
            inputs={"input": input_name},
            outputs=layer_outputs,
            num_channels=c,
            epsilon=node.get_attr("epsilon"),
            param_attr=string("{}_{}".format(node.name, gamma.name)),
            bias_attr=string("{}_{}".format(node.name, beta.name)),
            moving_mean_name=string("{}_{}".format(node.name,
                                                   moving_mean.name)),
            moving_variance_name=string("{}_{}".format(node.name,
                                                       moving_var.name)),
            is_test=True)

        if data_format == "NHWC":
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": node.name},
                outputs=[node.name],
                perm=[0, 2, 3, 1])

    def FusedBatchNormV3(self, node):
        self.FusedBatchNorm(node)

    def Mean(self, node):
        input = self.graph.get_input_node(node, 0)
        reduce_idx = self.graph.get_input_node(node, 1)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        dims = reduce_idx.value.tolist()
        keep_dims = node.get_attr("keep_dims")

        self.paddle_graph.add_layer(
            kernel="paddle.mean",
            inputs={"x": input.name},
            outputs=[node.name],
            axis=dims,
            keepdim=keep_dims)

    def Reshape(self, node):
        input = self.graph.get_input_node(node, 0)
        param = self.graph.get_input_node(node, 1)

        input_name = input.name

        if param.layer_type == "Const":
            shape = param.value.tolist()
            self.paddle_graph.add_layer(
                kernel="paddle.reshape",
                inputs={"x": input_name},
                outputs=[node.name],
                shape=shape)
        else:
            self.paddle_graph.add_layer(
                kernel="paddle.reshape",
                inputs={"x": input_name,
                        "shape": param.name},
                outputs=[node.name])
        if param.layer_type != "Const":
            out_shape = numpy.array(node.out_shapes[0])
            if (out_shape > 0).any():
                out_shape[out_shape < 0] = 0
                self.paddle_graph.add_layer(
                    kernel="paddle.reshape",
                    inputs={"x": node.name},
                    outputs=[node.name],
                    shape=out_shape.tolist())

    def Pad(self, node):
        input = self.graph.get_input_node(node, 0)
        paddings = self.graph.get_input_node(node, 1)
        assert paddings.layer_type == "Const", "Padding should be Const"
        paddings = paddings.value.flatten().tolist()

        constant_values = 0
        if len(node.layer.input) > 2:
            constant_values = self.graph.get_input_node(node, 2)
            assert constant_values.layer_type == "Const", "Padding should be Const"
            constant_values = constant_values.value

        if len(paddings) == 8 and sum(paddings[:2]) == 0 \
            and sum(paddings[-2:]) == 0:
            paddings = paddings[2:-2]
            self.paddle_graph.add_layer(
                kernel="paddle.nn.functional.pad",
                inputs={"x": input.name},
                outputs=[node.name],
                pad=paddings,
                value=constant_values,
                data_format=string('NHWC'))
        else:
            self.paddle_graph.add_layer(
                kernel="paddle.nn.functional.pad",
                inputs={"x": input.name},
                outputs=[node.name],
                pad=paddings,
                value=constant_values)

    def MirrorPad(self, node):
        self.Pad(node)

    def PadV2(self, node):
        self.Pad(node)

    def Squeeze(self, node):
        input = self.graph.get_input_node(node, 0)
        squeeze_dims = node.get_attr('squeeze_dims')
        self.paddle_graph.add_layer(
            kernel="paddle.squeeze",
            inputs={"x": input.name},
            outputs=[node.name],
            axis=squeeze_dims)

    def Shape(self, node):
        input = self.graph.get_input_node(node, 0)
        input_name = input.name
        self.paddle_graph.add_layer(
            kernel="paddle.shape",
            inputs={"input": input_name},
            outputs=[node.name])

    def Size(self, node):
        input = self.graph.get_input_node(node, 0)
        input_name = input.name
        self.paddle_graph.add_layer(
            kernel="paddle.shape",
            inputs={"input": input_name},
            outputs=[node.name])
        self.paddle_graph.add_layer(
            kernel="paddle.prod", inputs={"x": node.name}, outputs=[node.name])

    def Ceil(self, node):
        input = self.graph.get_input_node(node, 0)
        self.paddle_graph.add_layer(
            kernel="paddle.ceil", inputs={"x": input.name},
            outputs=[node.name])

    def ArgMax(self, node):
        input = self.graph.get_input_node(node, 0)
        axis = self.graph.get_input_node(node, 1)
        assert axis.layer_type == "Const", "ArgMax only support Const parameter"
        axis = axis.value
        self.paddle_graph.add_layer(
            kernel="paddle.argmax",
            inputs={"x": input.name},
            outputs=[node.name],
            axis=axis)

    def TopKV2(self, node):
        input = self.graph.get_input_node(node, 0)
        k = self.graph.get_input_node(node, 1)
        assert k.layer_type == "Const", "ArgMax only support Const parameter"
        k = k.value
        sort = node.get_attr('sorted')
        self.paddle_graph.add_layer(
            kernel="paddle.topk",
            inputs={"x": input.name},
            outputs=[node.name],
            k=k,
            sorted=sort)

    def MatMul(self, node):
        x = self.graph.get_input_node(node, 0)
        y = self.graph.get_input_node(node, 1)
        transpose_a = node.get_attr('transpose_a')
        transpose_b = node.get_attr('transpose_b')
        if transpose_a is None:
            transpose_a = node.get_attr('adj_x')
        if transpose_b is None:
            transpose_b = node.get_attr('adj_y')
        self.paddle_graph.add_layer(
            kernel="paddle.matmul",
            inputs={"x": x.name,
                    "y": y.name},
            outputs=[node.name],
            transpose_x=transpose_a,
            transpose_y=transpose_b)

    def BatchMatMul(self, node):
        return self.MatMul(node)

    def BatchMatMulV2(self, node):
        return self.MatMul(node)

    def DepthwiseConv2dNative(self, node):
        op_name = name_generator("conv", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        input = self.graph.get_input_node(node, 0)
        kernel = self.graph.get_input_node(node, 1)
        assert kernel.layer_type == "Const", "Kernel of DepthwiseConv2DNative should be Const"

        in_shape = input.out_shapes[0]
        k_size = kernel.out_shapes[0]
        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()

        kernel_weight_name = op_name + ".weight"
        self.params[kernel_weight_name] = numpy.transpose(kernel.value,
                                                          (2, 3, 0, 1))

        input_name = input.name
        if data_format == "NHWC":
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]
            transpose_name = gen_name('depthwise_conv2d', 'transpose')
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": input.name},
                outputs=[transpose_name],
                perm=[0, 3, 1, 2])
            input_name = transpose_name

        self.paddle_graph.add_layer(
            kernel="paddle.nn.Conv2D",
            inputs={"input": input_name},
            outputs=layer_outputs,
            weight_attr=string(kernel_weight_name),
            bias_attr=False,
            in_channels=in_shape[1],
            out_channels=k_size[2],
            kernel_size=k_size[0:2],
            stride=strides[2:4],
            dilation=dilations[2:4],
            groups=k_size[3] * in_shape[1],
            padding=string(pad_mode))

        if data_format == "NHWC":
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": node.name},
                outputs=[node.name],
                perm=[0, 2, 3, 1])

    def AvgPool(self, node):
        input = self.graph.get_input_node(node, 0)

        k_size = node.get_attr("ksize")
        strides = node.get_attr("strides")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()

        input_name = input.name
        if data_format == "NHWC":
            transpose_name = gen_name("avg_pool", "transpose")
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": input.name},
                outputs=[transpose_name],
                perm=[0, 3, 1, 2])
            strides = [strides[i] for i in [0, 3, 1, 2]]
            k_size = [k_size[i] for i in [0, 3, 1, 2]]
            input_name = transpose_name

        op_name = name_generator("pool", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]

        # TODO(syf): The op has diff.
        self.paddle_graph.add_layer(
            kernel="paddle.nn.AvgPool2D",
            inputs={"input": input_name},
            outputs=layer_outputs,
            kernel_size=k_size[2:4],
            stride=strides[2:4],
            padding=string(pad_mode))

        if data_format == "NHWC":
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": node.name},
                outputs=[node.name],
                perm=[0, 2, 3, 1])

    def Pack(self, node):
        inputs_list = list()
        for i in range(len(node.inputs)):
            inputs_list.append(self.graph.get_input_node(node, i))
        input_names = [i.name for i in inputs_list]
        axis = node.get_attr("axis")
        self.paddle_graph.add_layer(
            kernel="paddle.stack",
            inputs={"x": input_names},
            outputs=[node.name],
            axis=axis)
        if len(node.out_shapes[0]) == 1:
            self.paddle_graph.add_layer(
                kernel="paddle.reshape",
                inputs={"x": node.name},
                outputs=[node.name],
                shape=[-1])

    def Unpack(self, node):
        input = self.graph.get_input_node(node, 0)
        axis = node.get_attr("axis")
        num = node.get_attr("num")
        shape = input.out_shapes[0]
        input_name = input.name
        if len(shape) == 1:
            if shape[0] > 0 and num == shape[0]:
                self.paddle_graph.add_layer(
                    kernel="paddle.unsqueeze",
                    inputs={"x": input.name},
                    outputs=[node.name],
                    axis=[0])
                input_name = node.name
                axis = 1
            else:
                raise Exception("Unexpected situation happend in Unpack OP")
        layer_outputs = [
            "{}_p{}".format(node.layer_name, i) for i in range(num)
        ]
        if len(layer_outputs) == 1:
            layer_outputs[0] = "[{}]".format(node.layer_name)
        self.paddle_graph.add_layer(
            kernel="paddle.unstack",
            inputs={"x": input_name},
            outputs=layer_outputs,
            axis=axis,
            num=num)

    def ConcatV2(self, node):
        inputs_list = list()
        for i in range(len(node.inputs) - 1):
            inputs_list.append(self.graph.get_input_node(node, i))
        axis = self.graph.get_input_node(node, -1)
        assert axis.layer_type == "Const", "axis for ConcatV2 must be type Const"
        axis = axis.value
        if axis < 0:
            axis += len(inputs_list[0].out_shapes[0])

        input_names = [i.name for i in inputs_list]
        self.paddle_graph.add_layer(
            kernel="paddle.concat",
            inputs={"x": input_names},
            outputs=[node.name],
            axis=axis)

    def Concat(self, node):
        inputs_list = list()
        for i in range(1, len(node.inputs)):
            inputs_list.append(self.graph.get_input_node(node, i))
        axis = self.graph.get_input_node(node, 0)
        assert axis.layer_type == "Const", "axis for ConcatV2 must be type Const"
        axis = axis.value
        if axis < 0:
            axis += len(inputs_list[0].out_shapes[0])

        input_names = [i.name for i in inputs_list]
        self.paddle_graph.add_layer(
            kernel="paddle.concat",
            inputs={"x": input_names},
            outputs=[node.name],
            axis=axis)

    def AddN(self, node):
        inputs_list = list()
        for i in range(len(node.inputs) - 1):
            inputs_list.append(self.graph.get_input_node(node, i))

        input_names = [i.name for i in inputs_list]
        self.paddle_graph.add_layer(
            kernel="paddle.add_n",
            inputs={"inputs": input_names},
            outputs=[node.name])

    def StridedSlice(self, node):
        input = self.graph.get_input_node(node, 0)
        begin = self.graph.get_input_node(node, 1)
        end = self.graph.get_input_node(node, 2)
        strides = self.graph.get_input_node(node, 3)

        if strides.layer_type == "Const":
            strides = strides.value.tolist()
        else:
            strides = self.decoder.infer_tensor(strides)
        if begin.layer_type == "Const":
            begin = begin.value.tolist()
        else:
            begin = self.decoder.infer_tensor(begin)
        if end.layer_type == "Const":
            end = end.value.tolist()
        else:
            end = self.decoder.infer_tensor(end)

        assert len(set(strides)) == 1 and strides[
            0] == 1, "Only support strides be 1 in StridedSlice OP"

        if len(begin) < len(input.out_shapes[0]):
            begin = begin + [0] * (len(input.out_shapes[0]) - len(begin))
        if len(end) < len(input.out_shapes[0]):
            end = end + [0] * (len(input.out_shapes[0]) - len(end))
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

        if input.dtype == "bool":
            self.paddle_graph.add_layer(
                "paddle.cast",
                inputs={"x": input.name},
                outputs=[input.name],
                dtype=string("int32"))

        self.paddle_graph.add_layer(
            kernel="paddle.slice",
            inputs={"input": input.name},
            outputs=[node.name],
            axes=[i for i in range(len(new_begin))],
            starts=new_begin,
            ends=new_end)

        if input.dtype == "bool":
            self.paddle_graph.add_layer(
                "paddle.cast",
                inputs={"x": node.name},
                outputs=[node.name],
                dtype=string("bool"))

        if len(new_axes) > 0:
            self.paddle_graph.add_layer(
                kernel="paddle.unsqueeze",
                inputs={"x": node.name},
                outputs=[node.name],
                axis=new_axes)
        if len(shrink_axes) > 0:
            if len(input.out_shapes[0]) + len(new_axes) <= 1:
                pass
            else:
                self.paddle_graph.add_layer(
                    kernel="paddle.squeeze",
                    inputs={"x": node.name},
                    outputs=[node.name],
                    axis=shrink_axes)

    def Prod(self, node):
        input = self.graph.get_input_node(node, 0)
        reduction_indices = self.graph.get_input_node(node, 1)
        assert reduction_indices.layer_type == "Const"
        keep_dims = node.get_attr('keep_dims')
        axis = reduction_indices.value

        self.paddle_graph.add_layer(
            kernel="paddle.prod",
            inputs={"x": input.name},
            outputs=[node.layer_name],
            keepdim=keep_dims,
            axis=axis)

    def Split(self, node):
        dim = self.graph.get_input_node(node, 0)
        input = self.graph.get_input_node(node, 1)
        assert dim.layer_type == "Const"
        num_split = node.get_attr('num_split')
        dim = dim.value

        self.paddle_graph.add_layer(
            kernel="paddle.split",
            inputs={"x": input.name},
            outputs=[
                "{}_p{}".format(node.layer_name, i) for i in range(num_split)
            ],
            num_or_sections=num_split,
            axis=dim)

    def SplitV(self, node):
        input = self.graph.get_input_node(node, 0)
        size_splits = self.graph.get_input_node(node, 1)
        assert size_splits.layer_type == "Const", "size_splits of SplitV OP should be Const"
        size_splits = size_splits.value.tolist()
        dim = self.graph.get_input_node(node, 2)
        assert dim.layer_type == "Const", "dim of SplitV OP should be Const"
        dim = dim.value

        self.paddle_graph.add_layer(
            kernel="paddle.split",
            inputs={"x": input.name},
            outputs=[
                "{}_p{}".format(node.layer_name, i)
                for i in range(len(size_splits))
            ],
            num_or_sections=size_splits,
            axis=dim)

    def Slice(self, node):
        input = self.graph.get_input_node(node, 0)
        begin = self.graph.get_input_node(node, 1)
        size = self.graph.get_input_node(node, 2)

        inputs = {"x": input.name}
        attrs = {}
        if begin.layer_type == "Const":
            begin = begin.value.tolist()
            attrs['offsets'] = begin
        else:
            begin = self.decoder.infer_tensor(
                begin, use_diff_inputs=False).tolist()
            attrs['offsets'] = begin
        if size.layer_type == "Const":
            size = size.value.tolist()
            attrs['shape'] = size
        else:
            shape = size.out_shapes[0]
            reshape_name = gen_name("slice", "reshape")
            self.paddle_graph.add_layer(
                kernel="paddle.reshape",
                inputs={"x": size.name},
                outputs=[reshape_name],
                shape=shape)
            inputs['shape'] = reshape_name
        self.paddle_graph.add_layer(
            kernel="paddle.crop", inputs=inputs, outputs=[node.name], **attrs)

    def ResizeNearestNeighbor(self, node):
        input = self.graph.get_input_node(node, 0)
        resize_shape = self.graph.get_input_node(node, 1)
        data_format = "NHWC"
        inputs = {"x": input.name}
        attrs = {
            "align_corners": node.get_attr("align_corners"),
            "mode": string("nearest"),
            "align_mode": 1
        }

        if resize_shape.layer_type == "Const":
            resize_shape = resize_shape.value.tolist()
            attrs["size"] = resize_shape
        else:
            shape = resize_shape.out_shapes[0]
            reshape_name = gen_name("resize_nearest", "reshape")
            self.paddle_graph.add_layer(
                kernel="paddle.reshape",
                inputs={"x": resize_shape.name},
                outputs=[reshape_name],
                shape=shape)
            inputs["size"] = reshape_name

        if data_format == "NHWC":
            transpose_name = gen_name("resize_nearest", "reshape")
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": input.name},
                outputs=[transpose_name],
                perm=[0, 3, 1, 2])
            inputs["x"] = transpose_name

        self.paddle_graph.add_layer(
            kernel="paddle.nn.functional.interpolate",
            inputs=inputs,
            outputs=[node.name],
            **attrs)

        if data_format == "NHWC":
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": node.name},
                outputs=[node.name],
                perm=[0, 2, 3, 1])

    def ResizeBilinear(self, node):
        input = self.graph.get_input_node(node, 0)
        resize_shape = self.graph.get_input_node(node, 1)
        data_format = "NHWC"
        inputs = {"x": input.name}
        attrs = {
            "align_corners": node.get_attr("align_corners"),
            "mode": string("bilinear"),
            "align_mode": 1
        }

        if resize_shape.layer_type == "Const":
            resize_shape = resize_shape.value.tolist()
            attrs["size"] = resize_shape
        else:
            shape = resize_shape.out_shapes[0]
            reshape_name = gen_name("resize_bilinear", "reshape")
            self.paddle_graph.add_layer(
                kernel="paddle.reshape",
                inputs={"x": resize_shape.name},
                outputs=[reshape_name],
                shape=shape)
            inputs["size"] = reshape_name

        if data_format == "NHWC":
            transpose_name = gen_name("resize_bilinear", "reshape")
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": input.name},
                outputs=[transpose_name],
                perm=[0, 3, 1, 2])
            inputs["x"] = transpose_name

        self.paddle_graph.add_layer(
            kernel="paddle.nn.functional.interpolate",
            inputs=inputs,
            outputs=[node.name],
            **attrs)

        if data_format == "NHWC":
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": node.name},
                outputs=[node.name],
                perm=[0, 2, 3, 1])

    def Cast(self, node):
        input = self.graph.get_input_node(node, 0)
        dtype = node.dtype
        self.paddle_graph.add_layer(
            kernel="paddle.cast",
            inputs={"x": input.name},
            outputs=[node.name],
            dtype=string(dtype))

    def Sum(self, node):
        input = self.graph.get_input_node(node, 0)
        reduce_idx = self.graph.get_input_node(node, 1)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        keep_dims = node.get_attr("keep_dims")
        dim = reduce_idx.value.tolist()

        self.paddle_graph.add_layer(
            kernel="paddle.sum",
            inputs={"x": input.name},
            outputs=[node.name],
            axis=dim,
            keepdim=keep_dims)

    def Max(self, node):
        input = self.graph.get_input_node(node, 0)
        reduce_idx = self.graph.get_input_node(node, 1)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        keep_dims = node.get_attr("keep_dims")
        dim = reduce_idx.value.tolist()
        self.paddle_graph.add_layer(
            kernel="paddle.max",
            inputs={"x": input.name},
            outputs=[node.name],
            axis=dim,
            keepdim=keep_dims)

    def RandomUniform(self, node):
        shape = self.graph.get_input_node(node, 0)
        if shape.layer_type == "Const":
            shape = shape.value.tolist()
            self.paddle_graph.add_layer(
                kernel="paddle.uniform",
                inputs={},
                outputs=[node.name],
                shape=shape,
                min=0.0,
                max=0.9999)
        else:
            self.paddle_graph.add_layer(
                kernel="paddle.uniform",
                inputs={'shape': shape.name},
                outputs=[node.name],
                min=0.0,
                max=0.9999)

    def Conv2DBackpropInput(self, node):
        op_name = name_generator("conv", self.nn_name2id)
        output_name = node.name
        layer_outputs = [op_name, output_name]
        out_shape = self.graph.get_input_node(node, 0)
        kernel = self.graph.get_input_node(node, 1)
        input = self.graph.get_input_node(node, 2)

        assert kernel.layer_type == "Const", "Kernel of Conv2DBackpropInput should be Const"

        if out_shape.layer_type == "Const":
            out_shape = out_shape.value.tolist()
        else:
            out_shape = self.decoder.infer_tensor(
                out_shape, out_shape=node.out_shapes[0])

        in_shape = input.out_shapes[0]
        if in_shape.count(-1) > 2:
            in_shape = self.decoder.infer_tensor(
                input, use_diff_inputs=False).shape
        k_size = kernel.out_shapes[0]
        if k_size.count(-1) > 2:
            k_size = self.decoder.infer_tensor(
                kernel, use_diff_inputs=False).shape

        pad_mode = node.get_attr("padding").decode()
        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()

        kernel_name = op_name + ".weight"
        self.params[kernel_name] = numpy.transpose(kernel.value, (3, 2, 0, 1))

        input_name = input.name
        if data_format == "NHWC":
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]
            transpose_name = gen_name("conv2dbackpropinput", "transpose")
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": input.name},
                outputs=[transpose_name],
                perm=[0, 3, 1, 2])
            input_name = transpose_name

        self.paddle_graph.add_layer(
            "self.create_parameter",
            inputs={},
            outputs=["{}_{}".format(node.name, kernel_name).replace(".", "_")],
            shape=self.params[kernel_name].shape,
            attr=string(kernel_name))

        self.paddle_graph.add_layer(
            kernel="paddle.nn.functional.conv2d_transpose",
            inputs={
                "x": input_name,
                "weight":
                "{}_{}".format(node.name, kernel_name).replace(".", "_")
            },
            outputs=[node.name],
            bias=None,
            stride=strides[2:4],
            dilation=dilations[2:4],
            padding=string(pad_mode),
            output_size=out_shape[1:3])

        if data_format == "NHWC":
            self.paddle_graph.add_layer(
                kernel="paddle.transpose",
                inputs={"x": node.name},
                outputs=[node.name],
                perm=[0, 2, 3, 1])

    def Tile(self, node):
        input = self.graph.get_input_node(node, 0)
        repeat_times = self.graph.get_input_node(node, 1)
        inputs = {"x": input.name}
        attr = dict()
        in_shape = input.out_shapes[0]
        if repeat_times.layer_type == "Const":
            repeat_times = repeat_times.value.tolist()
            attr["repeat_times"] = repeat_times
        else:
            inputs["repeat_times"] = repeat_times.name

        self.paddle_graph.add_layer(
            kernel="paddle.tile", inputs=inputs, outputs=[node.name], **attr)

    def Range(self, node):
        start = self.graph.get_input_node(node, 0)
        limit = self.graph.get_input_node(node, 1)
        delta = self.graph.get_input_node(node, 2)
        inputs = dict()
        attr = dict()

        dtype = 'int32'
        if start.dtype.startswith('float'):
            dtype = start.dtype
        if start.layer_type == "Const":
            attr["start"] = start.value
        else:

            inputs["start"] = start.name
        if limit.dtype.startswith('float'):
            dtype = limit.dtype
        if limit.layer_type == "Const":
            attr["end"] = limit.value
        else:
            inputs["end"] = limit.name
        if delta.dtype.startswith('float'):
            dtype = delta.dtype
        if delta.layer_type == "Const":
            attr["step"] = delta.value
        else:
            inputs["step"] = delta.name
        node.set_dtype(dtype)
        attr["dtype"] = string(node.dtype)

        self.paddle_graph.add_layer(
            kernel="paddle.arange", inputs=inputs, outputs=[node.name], **attr)

    def SquaredDifference(self, node):
        x = self.graph.get_input_node(node, 0)
        y = self.graph.get_input_node(node, 1)
        inputs = {"x": x.name, "y": y.name}
        x_shape = x.out_shapes[0]
        y_shape = y.out_shapes[0]
        # TODO(syf)
        layer_id = self.paddle_graph.add_layer(
            "paddle.subtract", inputs=inputs, outputs=[node.name])
        self.paddle_graph.layers[layer_id].input_shapes = {
            "x": x_shape,
            "y": y_shape
        }

        inputs = {"x": node.name, "y": node.name}
        x_shape = node.out_shapes[0]
        y_shape = node.out_shapes[0]
        layer_id = self.paddle_graph.add_layer(
            "paddle.multiply", inputs=inputs, outputs=[node.name])
        self.paddle_graph.layers[layer_id].input_shapes = {
            "x": x_shape,
            "y": y_shape
        }

    def OneHot(self, node):
        input = self.graph.get_input_node(node, 0)
        depth = self.graph.get_input_node(node, 1)
        on_value = self.graph.get_input_node(node, 2)
        off_value = self.graph.get_input_node(node, 3)
        assert depth.layer_type == 'Const', 'Parameter depth should be Const in OneHot'
        assert on_value.layer_type == 'Const', 'Parameter on_value should be Const in OneHot'
        assert off_value.layer_type == 'Const', 'Parameter off_value should be Const in OneHot'

        attr = {'depth': depth.value}
        on_value = on_value.value
        off_value = off_value.value
        assert math.fabs(on_value -
                         1.0) < 1e-06, "on_value should be 1 in OneHot"
        assert math.fabs(off_value -
                         0.0) < 1e-06, "off_value should be 0 in OneHot"

        self.paddle_graph.add_layer(
            "paddle.nn.functional.one_hot",
            inputs={"x": input.name},
            outputs=[node.name],
            num_classes=depth.value)

    def Pow(self, node):
        x = self.graph.get_input_node(node, 0)
        factor = self.graph.get_input_node(node, 1)
        inputs = {"x": x.name}
        attr = dict()
        if factor.layer_type == 'Const':
            attr["y"] = factor.value.tolist()
        else:
            inputs["y"] = factor.name
        self.paddle_graph.add_layer(
            "paddle.pow", inputs=inputs, outputs=[node.name], **attr)

    def All(self, node):
        input = self.graph.get_input_node(node, 0)
        reduce_idx = self.graph.get_input_node(node, 1)
        assert reduce_idx.layer_type == "Const", "Only support Const parameter[reduce_idx]"
        attr = dict()
        attr["axis"] = reduce_idx.value.tolist()
        attr["keepdim"] = node.get_attr("keep_dims")

        input_name = input.name
        if input.dtype != "bool":
            input_name = gen_name("all", "cast")
            self.paddle_graph.add_layer(
                "paddle.cast",
                inputs={"x": input.name},
                outputs=[input_name],
                dtype=string("bool"))
        self.paddle_graph.add_layer(
            "paddle.all", inputs={"x": input_name}, outputs=[node.name], **attr)

        node.layer.attr['dtype'].type = 10

    def GatherV2(self, node):
        embeddings = self.graph.get_input_node(node, 0)
        index = self.graph.get_input_node(node, 1)
        axis = self.graph.get_input_node(node, 2)
        assert axis.layer_type == 'Const', "Only support Const parameter[axis]"
        axis = axis.value
        index_name = index.name
        if len(index.out_shapes[0]) != 1:
            reshape_name = gen_name("gather", "reshape")
            index_name = reshape_name
            self.paddle_graph.add_layer(
                "paddle.reshape",
                inputs={"x": index.name},
                outputs=[reshape_name],
                shape=[-1])
        inputs = {'x': embeddings.name, 'index': index_name}
        self.paddle_graph.add_layer(
            "paddle.gather", inputs=inputs, outputs=[node.name], axis=axis)
        if len(index.out_shapes[0]) != 1:
            out_shape = node.out_shapes[0]
            self.paddle_graph.add_layer(
                kernel="paddle.reshape",
                inputs={"x": node.name},
                outputs=[node.name],
                shape=out_shape)

    def GatherNd(self, node):
        x = self.graph.get_input_node(node, 0)
        index = self.graph.get_input_node(node, 1)
        inputs = {'x': x.name, 'index': index.name}
        self.paddle_graph.add_layer(
            "paddle.gather_nd", inputs=inputs, outputs=[node.name])

    def ExpandDims(self, node):
        x = self.graph.get_input_node(node, 0, copy=True)
        y = self.graph.get_input_node(node, 1, copy=True)
        inputs = {"x": x.name}
        attr = dict()
        if y.layer_type == 'Const':
            dim = y.value.tolist()
            if not isinstance(dim, list):
                dim = [dim]
            attr['axis'] = dim
        else:
            inputs['axis'] = y.name
        self.paddle_graph.add_layer(
            "paddle.unsqueeze", inputs=inputs, outputs=[node.name], **attr)

    def ReverseV2(self, node):
        x = self.graph.get_input_node(node, 0)
        axis = self.graph.get_input_node(node, 1)
        inputs = {"x": x.name}
        attr = dict()
        if axis.layer_type == 'Const':
            axis = axis.value.tolist()
            if not isinstance(axis, list):
                axis = [axis]
            attr['axis'] = axis
        else:
            inputs['axis'] = axis.name
        self.paddle_graph.add_layer(
            "paddle.flip", inputs=inputs, outputs=[node.name], **attr)

    def BatchToSpaceND(self, node):
        '''
        reshape->transpose->reshape->crop
        '''
        x = self.graph.get_input_node(node, 0)
        block_shape = self.graph.get_input_node(node, 1)
        crops = self.graph.get_input_node(node, 2)
        if block_shape.layer_type == "Const":
            block_shape = block_shape.value.tolist()
        if crops.layer_type == "Const":
            crops = crops.value.tolist()
        data_format = x.get_attr("data_format").decode()
        if data_format == "NHWC":
            n, h, w, c = x.out_shapes[0]
        else:
            n, c, h, w = x.out_shapes[0]
        input_name = x.name
        #reshape
        shape = block_shape + [-1, h, w, c]
        reshape_name = gen_name("batch_to_space", "reshape")
        self.paddle_graph.add_layer(
            kernel="paddle.reshape",
            inputs={"x": input_name},
            outputs=[reshape_name],
            shape=shape)
        #transpose
        perm = [len(block_shape)] + list(j for i in range(len(block_shape)) for j in (i + len(block_shape) + 1, i)) +\
                                    list(i + 2*len(block_shape) + 1 for i in range(len(x.out_shapes[0]) - len(block_shape) - 1))
        transpose_name = gen_name("batch_to_space", "transpose")
        self.paddle_graph.add_layer(
            kernel="paddle.transpose",
            inputs={"x": reshape_name},
            outputs=[transpose_name],
            perm=perm)
        #reshape
        shape = [-1] + list(i * j
                            for i, j in zip(block_shape, x.out_shapes[0][
                                1:])) + x.out_shapes[0][1 + len(block_shape):]
        reshape_name = gen_name("batch_to_space", "reshape")
        self.paddle_graph.add_layer(
            kernel="paddle.reshape",
            inputs={"x": transpose_name},
            outputs=[reshape_name],
            shape=shape)
        #crop
        attrs = {}
        crop_shape = shape
        crop_offsets = [0] * len(shape)
        for i in range(len(crops)):
            crop_shape[i + 1] = crop_shape[i + 1] - crops[i][0] - crops[i][1]
            crop_offsets[i + 1] = crops[i][0]
        attrs['shape'] = crop_shape
        attrs['offsets'] = crop_offsets
        self.paddle_graph.add_layer(
            kernel="paddle.crop",
            inputs={"x": reshape_name},
            outputs=[node.name],
            **attrs)

    def SpaceToBatchND(self, node):
        '''
        zero-pad->reshape->transpose->reshape
        '''
        x = self.graph.get_input_node(node, 0)
        block_shape = self.graph.get_input_node(node, 1)
        paddings = self.graph.get_input_node(node, 2)
        if block_shape.layer_type == "Const":
            block_shape = block_shape.value.tolist()
        if paddings.layer_type == "Const":
            paddings = paddings.value.flatten().tolist()
        input_name = x.name
        #zero-pad
        constant_values = 0
        pad_name = gen_name("space_to_batch", "pad")
        paddings = [0, 0] + paddings + [0, 0]
        self.paddle_graph.add_layer(
            kernel="paddle.nn.functional.pad",
            inputs={"x": input_name},
            outputs=[pad_name],
            pad=paddings,
            value=constant_values)
        #reshape
        n, h, w, c = x.out_shapes[0]
        h = h + paddings[2] + paddings[3]
        w = w + paddings[4] + paddings[5]
        shape = [
            n, h // block_shape[0], block_shape[0], w // block_shape[1],
            block_shape[1], c
        ]
        reshape_name = gen_name("space_to_batch", "reshape")
        self.paddle_graph.add_layer(
            kernel="paddle.reshape",
            inputs={"x": pad_name},
            outputs=[reshape_name],
            shape=shape)
        #transpose
        transpose_name = gen_name("space_to_batch", "transpose")
        self.paddle_graph.add_layer(
            kernel="paddle.transpose",
            inputs={"x": reshape_name},
            outputs=[transpose_name],
            perm=[2, 4, 0, 1, 3, 5])
        #reshape
        shape = [-1, h // block_shape[0], w // block_shape[1], c]
        self.paddle_graph.add_layer(
            kernel="paddle.reshape",
            inputs={"x": transpose_name},
            outputs=[node.name],
            shape=shape)
