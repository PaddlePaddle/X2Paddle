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

from x2paddle.parser.tf_parser import TFGraph
from x2paddle.core.emitter import Emitter
from x2paddle.core.fluid_code import FluidCode
from x2paddle.core.util import *
import numpy


class TFEmitter(Emitter):
    def __init__(self, parser):
        super(TFEmitter, self).__init__()
        self.parser = parser
        self.graph = parser.tf_graph
        # attr_node is used to record nodes that
        # only for define attribute of op
        self.attr_node = list()
        self.omit_nodes = list()
        self.weights = dict()

    def run(self):
        print("Total nodes: {}".format(len(self.graph.topo_sort)))
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if hasattr(self, op):
                emit_func = getattr(self, op)
                emit_func(node)

        for i in range(len(self.graph.topo_sort)):
            node_name = self.graph.topo_sort[i]
            if node_name in self.omit_nodes:
                continue
            node = self.graph.get_node(node_name)
            for layer in node.fluid_code.layers:
                print(layer.get_code())

        for name, param in self.weights.items():
            node = self.graph.get_node(name)
            export_paddle_param(param, node.layer_name.replace('/', '_'),
                                "params1")

    def Placeholder(self, node):
        shape = node.out_shapes[0]
        dtype = node.dtype
        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer("data",
                                  inputs=None,
                                  output=node,
                                  param_attr=attr)

    def Const(self, node):
        shape = node.out_shapes[0]
        dtype = node.dtype
        value = node.value
        initializer = "Constant(0.0)"
        if len(shape) == 0:
            assert value.size == 1, "Unexpected situation happend"
            shape = [1]
            initializer = "Constant({})".format(value)

        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name),
            'default_initializer': initializer
        }
        node.fluid_code.add_layer("create_parameter",
                                  inputs=None,
                                  output=node,
                                  param_attr=attr)
        self.weights[node.layer_name] = node.value

    def Transpose(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        perm = self.graph.get_node(node.layer.input[1], copy=True)
        assert perm.layer_type == "Const", "Perm of transpose OP should be Const"
        del self.weights[perm.layer_name]
        perm.fluid_code.clear()
        perm = perm.value.tolist()

        attr = {'perm': perm}
        node.fluid_code.add_layer("transpose",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def RealDiv(self, node):
        x = self.graph.get_node(node.layer.input[0], copy=True)
        y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {'x': x, 'y': y}
        node.fluid_code.add_layer("elementwise_div",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)

    def Relu(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("relu",
                                  inputs=input,
                                  output=node,
                                  param_attr=None)

    def Squeeze(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        squeeze_dims = node.get_attr('squeeze_dims')
        attr = {'axes': squeeze_dims}
        node.fluid_code.add_layer("squeeze",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def BiasAdd(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        bias = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {'x': input, 'y': bias}
        node.fluid_code.add_layer("elementwise_add",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)

    def Identity(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("assign",
                                  inputs=input,
                                  output=node,
                                  param_attr=None)

    def MaxPool(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        in_shape = input.out_shapes[0]
        k_size = node.get_attr("ksize")
        strides = node.get_attr("strides")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"

        if not channel_first:
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer("transpose",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]

        if pad_mode == "SAME":
            pad_h = get_same_padding(in_shape[2], k_size[0], strides[2])
            pad_w = get_same_padding(in_shape[3], k_size[1], strides[3])
            pad_h = pad_h[0] + pad_h[1]
            pad_w = pad_w[0] + pad_w[1]
            attr = {"paddings": [0, pad_h, 0, pad_w], "pad_value": -10000.0}
            if pad_h + pad_w != 0:
                node.fluid_code.add_layer(
                    "pad2d",
                    inputs=input if channel_first else node,
                    output=node,
                    param_attr=attr)
        attr = {
            "pool_size": k_size[1:3],
            "pool_type": string("max"),
            "pool_stride": strides[2:4]
        }
        node.fluid_code.add_layer("pool2d",
                                  inputs=input if channel_first else node,
                                  output=node,
                                  param_attr=attr)

        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer("transpose",
                                      inputs=node,
                                      output=node,
                                      param_attr=attr)

    def Conv2D(self, node):
        input = self.graph.get_node(node.layer.input[0], copy=True)
        kernel = self.graph.get_node(node.layer.input[1], copy=True)
        assert kernel.layer_type == "Const", "Kernel of Conv2D should be Const"
        self.omit_nodes.append(kernel.layer_name)

        in_shape = input.out_shapes[0]
        k_size = kernel.out_shapes[0]
        strides = node.get_attr("strides")
        dilations = node.get_attr("dilations")
        data_format = node.get_attr("data_format").decode()
        pad_mode = node.get_attr("padding").decode()
        channel_first = data_format == "NCHW"

        if not channel_first:
            self.weights[kernel.layer_name] = numpy.transpose(
                kernel.value, (3, 2, 0, 1))
            attr = {"perm": [0, 3, 1, 2]}
            node.fluid_code.add_layer("transpose",
                                      inputs=input,
                                      output=node,
                                      param_attr=attr)
            in_shape = [in_shape[i] for i in [0, 3, 1, 2]]
            strides = [strides[i] for i in [0, 3, 1, 2]]
            dilations = [dilations[i] for i in [0, 3, 1, 2]]

        if pad_mode == "SAME":
            pad_h = get_same_padding(in_shape[2], k_size[0], strides[2])
            pad_w = get_same_padding(in_shape[3], k_size[1], strides[3])
            attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}
            if pad_h[0] + pad_h[1] + pad_w[0] + pad_w[1] != 0:
                node.fluid_code.add_layer(
                    "pad2d",
                    inputs=input if channel_first else node,
                    output=node,
                    param_attr=attr)
        attr = {
            "bias_attr": False,
            "param_attr": string(kernel.layer_name),
            "num_filters": k_size[3],
            "filter_size": k_size[0:2],
            "stride": strides[2:4],
            "dilation": dilations[2:4]
        }
        node.fluid_code.add_layer("conv2d",
                                  inputs=input if channel_first else node,
                                  output=node,
                                  param_attr=attr)

        if not channel_first:
            attr = {"perm": [0, 2, 3, 1]}
            node.fluid_code.add_layer("transpose",
                                      inputs=node,
                                      output=node,
                                      param_attr=attr)
