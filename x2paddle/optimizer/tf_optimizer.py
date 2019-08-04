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

# TODO useless node remove
from x2paddle.op_mapper.tf_op_mapper import TFOpMapper
from x2paddle.core.util import *


class TFOptimizer(object):
    activation_ops = {
        'Relu': 'relu',
        'Sigmoid': 'sigmoid',
        'Relu6': 'relu6',
        'swish_f32': 'swish'
    }
    layers_with_act = [
        'Conv2D', 'BiasAdd', 'DepthwiseConv2dNative', 'Conv2DBackpropInput',
        'FusedBatchNorm'
    ]
    layers_with_bias = [
        'Conv2D', 'DepthwiseConv2dNative', 'Conv2DBackpropInput'
    ]

    def __init__(self, op_mapper):
        self.op_mapper = op_mapper
        self.graph = op_mapper.graph

    def delete_redundance_code(self):
        for node_name in self.graph.topo_sort:
            if node_name in self.op_mapper.omit_nodes:
                node = self.graph.get_node(node_name)
                omit_freq = self.op_mapper.omit_nodes.count(node_name)
                if len(node.outputs) <= omit_freq:
                    node.fluid_code.clear()

    # TODO activation merge
    def merge_activation(self):
        act_nodes = list()
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            if node.layer_type in self.activation_ops:
                act_nodes.append(node_name)

        for act_node_name in act_nodes:
            node = self.graph.get_node(act_node_name)
            input = self.graph.get_node(node.inputs[0])
            if input.layer_type not in self.layers_with_act:
                continue
            if len(input.fluid_code.layers) == 0:
                continue
            if 'act' in input.fluid_code.layers[
                    -1].param_attr and input.fluid_code.layers[-1].param_attr[
                        'act'] is not None:
                continue
            if len(input.outputs) != 1:
                continue
            input.fluid_code.layers[-1].param_attr['act'] = string(
                self.activation_ops[node.layer_type])
            input.fluid_code.layers[-1].output = node.fluid_code.layers[
                0].output
            self.graph.remove_node(act_node_name)

    # TODO bias merge
    def merge_bias(self):
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            if node.layer_type == "BiasAdd":
                input = self.graph.get_node(node.inputs[0])
                if input.layer_type not in self.layers_with_bias:
                    continue
                if len(input.outputs) != 1:
                    continue
                if len(input.fluid_code.layers) == 0:
                    continue
                bias_with_act = False
                if 'act' in node.fluid_code.layers[-1].param_attr:
                    bias_with_act = True
                layer_with_act = False
                if 'act' in input.fluid_code.layers[
                        -1].param_attr and input.fluid_code.layers[
                            -1].param_attr['act'] is not None:
                    layer_with_act = True

                if bias_with_act and layer_with_act:
                    continue
                if not input.fluid_code.layers[-1].param_attr['bias_attr']:
                    bias_name = node.inputs[1]
                    input.fluid_code.layers[-1].param_attr[
                        'bias_attr'] = string(bias_name)
                    input.fluid_code.layers[-1].output = node.fluid_code.layers[
                        0].output
                    if bias_with_act:
                        input.fluid_code.layers[-1].param_attr[
                            'act'] = node.fluid_code.layers[-1].param_attr[
                                'act']
                    node.fluid_code.clear()
