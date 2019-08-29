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
        'FusedBatchNorm', 'conv2d', 'elementwise_add', 'conv2d_transpose',
        'batch_norm'
    ]
    layers_with_bias = [
        'Conv2D', 'DepthwiseConv2dNative', 'Conv2DBackpropInput', 'conv2d',
        'conv2d_transpose'
    ]

    def __init__(self, op_mapper):
        self.op_mapper = op_mapper
        self.graph = op_mapper.graph

    def delete_redundance_code(self):
        for node_name in self.graph.topo_sort:
            if node_name in self.op_mapper.omit_nodes:
                node = self.graph.get_node(node_name)
                if node is None:
                    continue
                omit_freq = self.op_mapper.omit_nodes.count(node_name)
                if len(node.outputs) <= omit_freq:
                    node.fluid_code.clear()

                    # remove node from graph
                    input_names = node.inputs
                    output_names = node.outputs
                    for in_name in input_names:
                        in_node = self.graph.get_node(in_name)
                        index = in_node.outputs.index(node_name)
                        del in_node.outputs[index]
                    for out_name in output_names:
                        out_node = self.graph.get_node(out_name)
                        index = out_node.inputs.index(node_name)
                        del out_node.inputs[index]
                    del self.graph.node_map[node_name]

    def strip_graph(self):
        visited_nodes = set()

        def visit(node_name):
            if node_name in visited_nodes:
                return
            visited_nodes.add(node_name)
            input_names = self.graph.get_node(node_name).inputs
            for in_name in input_names:
                visit(in_name)

        for node_name in self.graph.output_nodes:
            visit(node_name)

        for i, node_name in enumerate(self.graph.topo_sort):
            if node_name not in visited_nodes:
                node = self.graph.get_node(node_name)
                if node is None:
                    continue
                input_names = node.inputs
                output_names = node.outputs
                for in_name in input_names:
                    in_node = self.graph.get_node(in_name)
                    index = in_node.outputs.index(node_name)
                    del in_node.outputs[index]
                for out_name in output_names:
                    out_node = self.graph.get_node(out_name)
                    index = out_node.inputs.index(node_name)
                    del out_node.inputs[index]
                del self.graph.node_map[node_name]

    # TODO activation merge
    def merge_activation(self):
        act_nodes = list()
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            if node is None:
                continue
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
            index = -1
            for i in range(len(input.fluid_code.layers)):
                if input.fluid_code.layers[i].op in self.layers_with_act:
                    index = i
                    break
            input.fluid_code.layers[index].param_attr['act'] = string(
                self.activation_ops[node.layer_type])
            input.fluid_code.layers[-1].output = node.fluid_code.layers[
                0].output
            self.graph.remove_node(act_node_name)

    # TODO bias merge
    def merge_bias(self):
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            if node is None:
                continue
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
                index = -1
                for i in range(len(input.fluid_code.layers)):
                    if input.fluid_code.layers[i].op in self.layers_with_bias:
                        index = i
                        break
                if 'act' in input.fluid_code.layers[
                        index].param_attr and input.fluid_code.layers[
                            index].param_attr['act'] is not None:
                    layer_with_act = True

                if bias_with_act and layer_with_act:
                    continue
                if not input.fluid_code.layers[index].param_attr['bias_attr']:
                    bias_name = node.inputs[1]
                    input.fluid_code.layers[index].param_attr[
                        'bias_attr'] = string(bias_name)
                    input.fluid_code.layers[-1].output = node.fluid_code.layers[
                        0].output
                    if bias_with_act:
                        input.fluid_code.layers[index].param_attr[
                            'act'] = node.fluid_code.layers[-1].param_attr[
                                'act']
                    node.fluid_code.clear()
                    self.graph.remove_node(node.layer_name)

    def remove_transpose(self):
        optimize_ops = [
            'Conv2D', 'MaxPool', 'FusedBatchNorm', 'DepthwiseConv2dNative',
            'AvgPool', 'Pad', 'Conv2DBackpropInput', 'ResizeNearestNeighbor',
            'ResizeBilinear'
        ]
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            if node is None:
                continue
            if node.layer_type not in optimize_ops:
                continue
            if node.fluid_code.layers[
                    -1].op != "transpose" or node.fluid_code.layers[
                        -1].param_attr["perm"] != [0, 2, 3, 1]:
                continue
            output_names = node.outputs
            can_be_removed = True
            for out_name in output_names:
                out_node = self.graph.get_node(out_name)
                if out_node.layer_type == "BiasAdd":
                    can_be_removed = True
                if out_node.fluid_code.layers[
                        0].op != "transpose" or out_node.fluid_code.layers[
                            0].param_attr["perm"] != [0, 3, 1, 2]:
                    can_be_removed = False
                    break

            if can_be_removed and len(output_names) > 0:
                last_out = node.fluid_code.layers[-1].inputs
                del node.fluid_code.layers[-1]
                for out_name in output_names:
                    out_node = self.graph.get_node(out_name)
                    if out_node.layer_type == "BiasAdd":
                        del out_node.fluid_code.layers[0]
                        out_node.fluid_code.layers[0].inputs['x'] = last_out
                    else:
                        del out_node.fluid_code.layers[0]
                        out_node.fluid_code.layers[0].inputs = last_out
