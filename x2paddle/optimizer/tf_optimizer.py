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
from x2paddle.core.fluid_code import Layer
from x2paddle.core.util import *
import six
import numpy
import copy as cp


def exist_act(node):
    for layer in node.fluid_code.layers:
        if layer.param_attr is not None:
            act = layer.param_attr.get("act", None)
            if act is not None:
                return True
    return False


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

    def optimize_elementwise_op(self):
        elementwise_ops = [
            'Sub', 'Add', 'RealDiv', 'Maximum', 'Mul', 'FloorDiv',
            'GreaterEqual'
        ]
        revertable_ops = ['Add', 'Mul']
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            if node is None:
                continue
            if node.layer_type in elementwise_ops:
                if len(node.fluid_code.layers) != 2:
                    continue
                if node.fluid_code.layers[0].op != "expand":
                    continue
                expand_out = node.fluid_code.layers[0].output
                expand_in = node.fluid_code.layers[0].inputs
                expand_times = node.fluid_code.layers[0].param_attr[
                    "expand_times"]

                x = node.fluid_code.layers[1].inputs["x"]
                y = node.fluid_code.layers[1].inputs["y"]
                if isinstance(
                        x,
                        six.string_types) and node.layer_type in revertable_ops:
                    node.fluid_code.layers[1].inputs["y"] = x
                    node.fluid_code.layers[1].inputs["x"] = y
                    x = node.fluid_code.layers[1].inputs["x"]
                    y = expand_in
                elif isinstance(y, six.string_types):
                    y = expand_in
                else:
                    continue

                x_shape = x.out_shapes[0]
                y_shape = y.out_shapes[0]
                if len(x_shape) != len(y_shape):
                    continue
                if len(x_shape) == 4:
                    x_shape = [x_shape[i] for i in [0, 3, 1, 2]]
                    y_shape = [y_shape[i] for i in [0, 3, 1, 2]]

                continue_flag = True
                for i in range(len(x_shape)):
                    if y_shape[-1 * (i + 1)] == 1 and continue_flag:
                        expand_times[-1 * (i + 1)] = 1
                    else:
                        continue_flag = False

                if expand_times.count(1) == len(expand_times):
                    node.fluid_code.layers[1].inputs["y"] = expand_in
                    del node.fluid_code.layers[0]

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
                    self.graph.identity_map[node.layer_name] = input.layer_name

    def remove_transpose(self):
        graph_copy = cp.deepcopy(self.graph)
        nhwc_insensitive_ops = [
            'Relu', 'Relu6', 'Abs', 'Sigmoid', 'Exp', 'Rsqrt', 'swish_f32',
            'LeakyRelu', 'Cast', 'Tanh'
        ]
        elementwise_ops = [
            'Sub', 'Add', 'RealDiv', 'Maximum', 'Mul', 'FloorDiv',
            'GreaterEqual'
        ]
        optimize_ops = [
            'Conv2D', 'MaxPool', 'FusedBatchNorm', 'DepthwiseConv2dNative',
            'AvgPool', 'Pad', 'Conv2DBackpropInput', 'ResizeNearestNeighbor',
            'ResizeBilinear', "Placeholder"
        ]
        can_be_optimized_ops = [
            'Conv2D', 'MaxPool', 'FusedBatchNorm', 'DepthwiseConv2dNative',
            'AvgPool', 'Pad', 'Conv2DBackpropInput', 'ResizeNearestNeighbor',
            'ResizeBilinear', "Placeholder", 'Relu', 'Relu6', 'Abs', 'Sigmoid',
            'Exp', 'Rsqrt', 'swish_f32', 'LeakyRelu', 'Cast', 'Tanh'
        ]

        for node_name in self.graph.topo_sort:
            node = graph_copy.get_node(node_name)
            if node is None:
                continue
            if node.layer_type in can_be_optimized_ops:
                if node.fluid_code.layers[
                        -1].op != "transpose" or node.fluid_code.layers[
                            -1].param_attr["perm"] != [0, 2, 3, 1]:
                    continue
                can_be_removed = True
                output_names = node.outputs
                for out_name in output_names:
                    out_node = graph_copy.get_node(out_name)
                    if hasattr(out_node, "can_be_removed"):
                        if not out_node.can_be_removed:
                            can_be_removed = False
                            break
                    elif out_node.fluid_code.layers[
                            0].op != "transpose" or out_node.fluid_code.layers[
                                0].param_attr["perm"] != [0, 3, 1, 2]:
                        can_be_removed = False
                        break
                    elif out_node.layer_type in elementwise_ops:
                        can_be_removed = False
                        break
                if can_be_removed and len(node.fluid_code.layers) > 1:
                    true_node = self.graph.get_node(node_name)
                    if true_node.layer_type == "Placeholder":
                        index = self.graph.input_nodes.index(
                            true_node.fluid_code.layers[-2].output)
                        if isinstance(true_node.fluid_code.layers[-1].output,
                                      str):
                            self.graph.input_nodes[
                                index] = true_node.fluid_code.layers[-1].output
                        else:
                            self.graph.input_nodes[
                                index] = true_node.fluid_code.layers[
                                    -1].output.layer_name
                    true_node.fluid_code.layers[
                        -2].output = true_node.fluid_code.layers[-1].output
                    node.removed = True
                    del true_node.fluid_code.layers[-1]
                    for out_name in output_names:
                        out_node = self.graph.get_node(out_name)
                        out_node.fluid_code.layers[
                            1].inputs = out_node.fluid_code.layers[0].inputs
                        del out_node.fluid_code.layers[0]

        for node_name in self.graph.topo_sort:
            node = graph_copy.get_node(node_name)
            if node is None:
                continue
            if node.layer_type in elementwise_ops:
                can_be_removed = True
                if node.fluid_code.layers[
                        -1].op != "transpose" or node.fluid_code.layers[
                            -1].param_attr["perm"] != [0, 2, 3, 1]:
                    continue
                can_be_removed = True

                output_names = node.outputs
                for out_name in output_names:
                    out_node = graph_copy.get_node(out_name)
                    if len(out_node.fluid_code.layers) < 3:
                        can_be_removed = False
                        break
                    if hasattr(out_node, "can_be_removed"):
                        if not out_node.can_be_removed:
                            can_be_removed = False
                            break
                    if out_node.layer_type in can_be_optimized_ops:
                        if out_node.fluid_code.layers[
                                0].op != "transpose" or out_node.fluid_code.layers[
                                    0].param_attr["perm"] != [0, 3, 1, 2]:
                            can_be_removed = False
                            break
                    elif out_node.layer_type in elementwise_ops:
                        if out_node.fluid_code.layers[
                                0].op != "transpose" and out_node.fluid_code.layers[
                                    1].op != "transpose":
                            can_be_removed = False
                            break
                        if out_node.fluid_code.layers[0].op == "transpose":
                            if out_node.fluid_code.layers[0].param_attr[
                                    "perm"] != [0, 3, 1, 2]:
                                can_be_removed = False
                                break
                        if out_node.fluid_code.layers[1].op == "transpose":
                            if out_node.fluid_code.layers[1].param_attr[
                                    "perm"] != [0, 3, 1, 2]:
                                can_be_removed = False
                                break

                if can_be_removed and len(node.fluid_code.layers) > 1:
                    true_node = self.graph.get_node(node_name)
                    true_node.fluid_code.layers[
                        -2].output = true_node.fluid_code.layers[-1].output
                    del true_node.fluid_code.layers[-1]
                    for out_name in output_names:
                        out_node = self.graph.get_node(out_name)
                        if out_node.layer_type in can_be_optimized_ops:
                            out_node.fluid_code.layers[
                                1].inputs = out_node.fluid_code.layers[0].inputs
                            del out_node.fluid_code.layers[0]
                        elif out_node.layer_type in elementwise_ops:
                            if out_node.inputs[0] in node.layer_name:
                                if out_node.fluid_code.layers[
                                        1].op == 'transpose':
                                    out_node.fluid_code.layers[2].inputs[
                                        'x'] = out_node.fluid_code.layers[
                                            0].inputs
                                    del out_node.fluid_code.layers[0]
                                else:
                                    out_node.fluid_code.layers[1].inputs[
                                        'x'] = out_node.fluid_code.layers[
                                            0].inputs
                                    del out_node.fluid_code.layers[0]
                            elif out_node.inputs[1] in node.layer_name:
                                if out_node.fluid_code.layers[
                                        1].op == 'transpose':
                                    out_node.fluid_code.layers[2].inputs[
                                        'y'] = out_node.fluid_code.layers[
                                            1].inputs
                                    del out_node.fluid_code.layers[1]
                                else:
                                    out_node.fluid_code.layers[1].inputs[
                                        'y'] = out_node.fluid_code.layers[
                                            0].inputs
                                    del out_node.fluid_code.layers[0]
        graph_copy = cp.deepcopy(self.graph)
        for node_name in self.graph.topo_sort:
            node = graph_copy.get_node(node_name)
            if node is None or len(node.fluid_code.layers) < 2:
                continue
            if node.layer_type in can_be_optimized_ops and node.layer_type != "Placeholder":
                if node.fluid_code.layers[
                        -1].op != "transpose" or node.fluid_code.layers[
                            -1].param_attr["perm"] != [0, 2, 3, 1]:
                    continue
                can_be_removed = True
                output_names = node.outputs
                for out_name in output_names:
                    out_node = graph_copy.get_node(out_name)
                    if hasattr(out_node, "can_be_removed"):
                        if not out_node.can_be_removed:
                            can_be_removed = False
                            break
                    if len(out_node.fluid_code.layers) < 2:
                        can_be_removed = False
                        break
                    if out_node.layer_type in can_be_optimized_ops:
                        if out_node.fluid_code.layers[
                                0].op != "transpose" or out_node.fluid_code.layers[
                                    0].param_attr["perm"] != [0, 3, 1, 2]:
                            can_be_removed = False
                            break
                    elif out_node.layer_type in elementwise_ops:
                        if out_node.fluid_code.layers[
                                0].op != "transpose" and out_node.fluid_code.layers[
                                    1].op != "transpose":
                            can_be_removed = False
                            break
                        if out_node.fluid_code.layers[
                                0].op == "expand" or out_node.fluid_code.layers[
                                    1].op == "expand":
                            can_be_removed = False
                            break
                        if out_node.fluid_code.layers[0].op == "transpose":
                            if out_node.fluid_code.layers[0].param_attr[
                                    "perm"] != [0, 3, 1, 2]:
                                can_be_removed = False
                                break
                        if out_node.fluid_code.layers[1].op == "transpose":
                            if out_node.fluid_code.layers[1].param_attr[
                                    "perm"] != [0, 3, 1, 2]:
                                can_be_removed = False
                                break
                    elif out_node.layer_type not in elementwise_ops and out_node.layer_type not in can_be_optimized_ops:
                        can_be_removed = False
                        break

                if can_be_removed:
                    true_node = self.graph.get_node(node_name)
                    if len(true_node.fluid_code.layers) < 2:
                        continue
                    true_node.fluid_code.layers[
                        -2].output = true_node.fluid_code.layers[-1].output
                    del true_node.fluid_code.layers[-1]
                    for out_name in output_names:
                        out_node = self.graph.get_node(out_name)
                        if out_node.layer_type in can_be_optimized_ops:
                            out_node.fluid_code.layers[
                                1].inputs = out_node.fluid_code.layers[0].inputs
                            del out_node.fluid_code.layers[0]
                        elif out_node.layer_type in elementwise_ops:
                            if out_node.inputs[0] in node.layer_name:
                                if out_node.fluid_code.layers[
                                        1].op == 'transpose':
                                    if out_node.fluid_code.layers[
                                            2].op == 'transpose':
                                        out_node.fluid_code.layers[3].inputs[
                                            'x'] = out_node.fluid_code.layers[
                                                0].inputs
                                    else:
                                        out_node.fluid_code.layers[2].inputs[
                                            'x'] = out_node.fluid_code.layers[
                                                0].inputs
                                    del out_node.fluid_code.layers[0]
                                else:
                                    out_node.fluid_code.layers[1].inputs[
                                        'x'] = out_node.fluid_code.layers[
                                            0].inputs
                                    del out_node.fluid_code.layers[0]
                            elif out_node.inputs[1] in node.layer_name:
                                if out_node.fluid_code.layers[
                                        1].op == 'transpose':
                                    out_node.fluid_code.layers[2].inputs[
                                        'y'] = out_node.fluid_code.layers[
                                            1].inputs
                                    del out_node.fluid_code.layers[1]
                                else:
                                    out_node.fluid_code.layers[1].inputs[
                                        'y'] = out_node.fluid_code.layers[
                                            0].inputs
                                    del out_node.fluid_code.layers[0]

        graph_copy = cp.deepcopy(self.graph)
        for node_name in self.graph.topo_sort:
            node = graph_copy.get_node(node_name)
            if node is None:
                continue
            if node.layer_type in elementwise_ops:
                can_be_removed = True
                if len(node.fluid_code.layers) < 3:
                    continue

                numTranspose = 0
                numNotTranspose = 0

                for i in range(len(node.fluid_code.layers)):
                    if node.fluid_code.layers[i].op == 'transpose':
                        numTranspose += 1
                    elif node.fluid_code.layers[i].op != 'expand':
                        numNotTranspose += 1
                if numTranspose > numNotTranspose:
                    if node.fluid_code.layers[0].op == 'expand':
                        if node.fluid_code.layers[
                                1].op != 'transpose' or node.fluid_code.layers[
                                    2].op != 'transpose':
                            continue
                        else:
                            true_node = self.graph.get_node(node_name)
                            true_node.fluid_code.layers[3].inputs[
                                'x'] = true_node.fluid_code.layers[1].inputs
                            true_node.fluid_code.layers[3].inputs[
                                'y'] = true_node.fluid_code.layers[2].inputs

                            l = Layer()
                            l.op = 'transpose'
                            l.inputs = true_node.fluid_code.layers[3].output
                            l.param_attr = {'perm': [0, 3, 1, 2]}
                            if isinstance(l.inputs, six.string_types):
                                l.output = l.inputs
                            else:
                                l.output = l.inputs.layer_name
                            true_node.fluid_code.layers.append(l)
                            del true_node.fluid_code.layers[1]
                            del true_node.fluid_code.layers[1]
                    else:
                        if node.fluid_code.layers[
                                0].op != 'transpose' or node.fluid_code.layers[
                                    1].op != 'transpose':
                            continue
                        else:
                            true_node = self.graph.get_node(node_name)
                            true_node.fluid_code.layers[2].inputs[
                                'x'] = true_node.fluid_code.layers[0].inputs
                            true_node.fluid_code.layers[2].inputs[
                                'y'] = true_node.fluid_code.layers[1].inputs

                            l = Layer()
                            l.op = 'transpose'
                            l.inputs = true_node.fluid_code.layers[2].output
                            l.param_attr = {'perm': [0, 3, 1, 2]}
                            l.output = l.inputs.layer_name
                            true_node.fluid_code.layers.append(l)
                            del true_node.fluid_code.layers[0]
                            del true_node.fluid_code.layers[0]

    def make_nchw_input_output(self):
        for i, name in enumerate(self.graph.input_nodes):
            node = self.graph.get_node(name)
            if len(node.out_shapes[0]) == 4 and node.tf_data_format == "NHWC":
                shape = node.fluid_code.layers[0].param_attr["shape"]
                shape = [shape[j] for j in [0, 3, 1, 2]]
                node.fluid_code.layers[0].param_attr["shape"] = shape
                node.fluid_code.layers[0].output = "nhwc_" + name
                attr = {"perm": [0, 2, 3, 1]}
                node.fluid_code.add_layer("transpose",
                                          inputs="nhwc_" + name,
                                          output=node,
                                          param_attr=attr)
                self.graph.input_nodes[i] = "nhwc_" + name
        for i, name in enumerate(self.graph.output_nodes):
            node = self.graph.get_node(name)
            if node.layer_type != "transpose":
                if node.fluid_code.layers[-1].op == "transpose":
                    node.fluid_code.layers[-2].output = name
                    del node.fluid_code.layers[-1]

    def optimize_sub_graph(self):
        self.merge_batch_norm()
        self.merge_prelu()
        self.merge_scale()
        self.merge_affine_channel()

    def merge_batch_norm(self):
        for i, name in enumerate(self.graph.topo_sort):
            node = self.graph.get_node(name)
            if node is None:
                continue
            is_batch_norm = True
            if node.layer_type == "Add":
                in_nodes0 = [
                    self.graph.get_node(in_name) for in_name in node.inputs
                ]
                if in_nodes0[0].layer_type != "Mul" or in_nodes0[
                        1].layer_type != "Sub":
                    is_batch_norm = False
                    continue

                if exist_act(in_nodes0[0]) or exist_act(in_nodes0[1]):
                    is_batch_norm = False
                    continue

                in_nodes1 = [
                    self.graph.get_node(in_name)
                    for in_name in in_nodes0[0].inputs
                ]
                in_nodes2 = [
                    self.graph.get_node(in_name)
                    for in_name in in_nodes0[1].inputs
                ]
                if len(in_nodes1[0].out_shapes[0]) != 4:
                    is_batch_norm = False
                    continue
                if in_nodes1[1].layer_type != "Mul":
                    is_batch_norm = False
                    continue
                if exist_act(in_nodes1[1]):
                    is_batch_norm = False
                    continue

                if in_nodes2[0].layer_type != "Const" or in_nodes2[
                        1].layer_type != "Mul":
                    is_batch_norm = False
                    continue
                if exist_act(in_nodes2[1]):
                    is_batch_norm = False
                    continue

                in_nodes3 = [
                    self.graph.get_node(in_name)
                    for in_name in in_nodes1[1].inputs
                ]
                if in_nodes3[0].layer_type != "Rsqrt" or in_nodes3[
                        1].layer_type != "Const":
                    is_batch_norm = False
                    continue

                in_nodes4 = [
                    self.graph.get_node(in_name)
                    for in_name in in_nodes2[1].inputs
                ]
                if in_nodes4[0].layer_type != "Const" or in_nodes4[
                        1].layer_name != in_nodes1[1].layer_name:
                    is_batch_norm = False
                    continue

                in_nodes5 = self.graph.get_node(in_nodes3[0].inputs[0])
                if in_nodes5.layer_type != "Add":
                    is_batch_norm = False
                    continue
                if exist_act(in_nodes5):
                    is_batch_norm = False
                    continue

                in_nodes6 = [
                    self.graph.get_node(in_name) for in_name in in_nodes5.inputs
                ]
                if in_nodes6[0].layer_type != "Const" or in_nodes6[
                        1].layer_type != "Const":
                    is_batch_norm = False
                    continue

                if len(in_nodes0[0].outputs) != 1:
                    is_batch_norm = False
                    continue
                if len(in_nodes0[1].outputs) != 1:
                    is_batch_norm = False
                    continue
                if len(in_nodes1[1].outputs) != 2:
                    is_batch_norm = False
                    continue
                if len(in_nodes2[0].outputs) != 1:
                    is_batch_norm = False
                    continue
                if len(in_nodes2[1].outputs) != 1:
                    is_batch_norm = False
                    continue
                if len(in_nodes3[0].outputs) != 1:
                    is_batch_norm = False
                    continue
                if len(in_nodes3[1].outputs) != 1:
                    is_batch_norm = False
                    continue
                if len(in_nodes4[0].outputs) != 1:
                    is_batch_norm = False
                    continue
                if len(in_nodes5.outputs) != 1:
                    is_batch_norm = False
                    continue
                if len(in_nodes6[0].outputs) != 1:
                    is_batch_norm = False
                    continue
                if len(in_nodes6[1].outputs) != 1:
                    is_batch_norm = False
                    continue

                conv_shape = in_nodes1[0].out_shapes[0]
                if conv_shape[3] < 0:
                    is_batch_norm = False
                    continue

                # moving_variance
                if in_nodes6[0].value.size != conv_shape[3]:
                    is_batch_norm = False
                    continue

                # epsilon
                if in_nodes6[1].value.size != 1:
                    is_batch_norm = False
                    continue

                # gamma
                if in_nodes3[1].value.size != conv_shape[3]:
                    is_batch_norm = False
                    continue

                # moving_mean
                if in_nodes4[0].value.size != conv_shape[3]:
                    is_batch_norm = False
                    continue

                # beta
                if in_nodes2[0].value.size != conv_shape[3]:
                    is_batch_norm = False
                    continue

                if is_batch_norm:
                    index = in_nodes1[0].outputs.index(in_nodes0[0].layer_name)
                    in_nodes1[0].outputs[index] = node.layer_name
                    node.layer_type = "FusedBatchNorm"
                    node.inputs = [in_nodes1[0].layer_name]
                    act = node.fluid_code.layers[-1].param_attr.get("act", None)
                    node.fluid_code.clear()
                    attr = {
                        "epsilon": in_nodes6[1].value,
                        "param_attr": string(in_nodes3[1].layer_name),
                        "bias_attr": string(in_nodes2[0].layer_name),
                        "moving_mean_name": string(in_nodes4[0].layer_name),
                        "moving_variance_name": string(in_nodes6[0].layer_name),
                        "is_test": True,
                        "act": act
                    }

                    node.fluid_code.add_layer(
                        "batch_norm",
                        inputs=in_nodes1[0].fluid_code.layers[-1].output,
                        output=node,
                        param_attr=attr)

                del self.graph.node_map[in_nodes0[0].layer_name]
                del self.graph.node_map[in_nodes0[1].layer_name]
                del self.graph.node_map[in_nodes1[1].layer_name]
                del self.graph.node_map[in_nodes2[1].layer_name]
                del self.graph.node_map[in_nodes3[0].layer_name]
                del self.graph.node_map[in_nodes4[0].layer_name]
                del self.graph.node_map[in_nodes5.layer_name]

    def merge_prelu(self):
        for i, name in enumerate(self.graph.topo_sort):
            node = self.graph.get_node(name)
            if node is None:
                continue
            is_prelu = True
            if node.layer_type == "Add":
                if exist_act(node):
                    is_prelu = False
                    continue
                in_nodes0 = [
                    self.graph.get_node(in_name) for in_name in node.inputs
                ]
                if in_nodes0[0].layer_type != "Relu" or in_nodes0[
                        1].layer_type != "Mul":
                    is_prelu = False
                    continue
                if exist_act(in_nodes0[1]):
                    is_prelu = False
                    continue

                if len(in_nodes0[0].outputs) != 1 or len(
                        in_nodes0[1].outputs) != 1:
                    is_prelu = False
                    continue

                in_nodes1 = self.graph.get_node(in_nodes0[0].inputs[0])
                in_nodes2 = [
                    self.graph.get_node(in_name)
                    for in_name in in_nodes0[1].inputs
                ]
                if in_nodes2[1].layer_type != "Const" or numpy.fabs(
                        in_nodes2[1].value - 0.5) > 1e-06:
                    is_prelu = False
                    continue
                if in_nodes2[0].layer_type != "Mul":
                    is_prelu = False
                    continue
                if exist_act(in_nodes2[0]):
                    is_prelu = False
                    continue
                if len(in_nodes2[1].outputs) != 1 or len(
                        in_nodes2[0].outputs) != 1:
                    is_prelu = False
                    continue

                in_nodes3 = [
                    self.graph.get_node(in_name)
                    for in_name in in_nodes2[0].inputs
                ]
                if in_nodes3[0].layer_type != "Const" or in_nodes3[
                        1].layer_type != "Sub":
                    is_prelu = False
                    continue
                if exist_act(in_nodes3[1]):
                    is_prelu = False
                    continue
                if len(in_nodes3[0].outputs) != 1 or len(
                        in_nodes3[1].outputs) != 1:
                    is_prelu = False
                    continue

                in_nodes4 = [
                    self.graph.get_node(in_name)
                    for in_name in in_nodes3[1].inputs
                ]
                if in_nodes4[0].layer_name != in_nodes1.layer_name or in_nodes4[
                        1].layer_type != "Abs":
                    is_prelu = False
                    continue
                if len(in_nodes4[1].outputs) != 1:
                    is_prelu = False
                    continue

                in_nodes5 = self.graph.get_node(in_nodes4[1].inputs[0])
                if in_nodes5.layer_name != in_nodes1.layer_name:
                    is_prelu = False
                    continue

                if len(in_nodes0[0].outputs) != 1:
                    is_prelu = false
                    continue
                if len(in_nodes0[1].outputs) != 1:
                    is_prelu = False
                    continue
                if len(in_nodes1.outputs) < 3:
                    is_prelu = False
                    continue
                if len(in_nodes2[0].outputs) != 1:
                    is_prelu = false
                    continue
                if len(in_nodes2[1].outputs) != 1:
                    is_prelu = False
                    continue
                if len(in_nodes3[0].outputs) != 1:
                    is_prelu = False
                    continue
                if len(in_nodes3[1].outputs) != 1:
                    is_prelu = false
                    continue
                if len(in_nodes4[1].outputs) != 1:
                    is_prelu = False
                    continue

                mode = None
                in_shape = in_nodes1.out_shapes[0]
                if in_shape == list(in_nodes3[0].value.shape):
                    mode = "element"
                elif len(in_nodes3[0].value.shape) == 0:
                    mode = "all"
                elif len(in_nodes3[0].value.shape
                         ) == 1 and in_nodes3[0].value.shape[0] == 1:
                    mode = "all"
                elif len(in_shape) == 4 and len(
                        in_nodes3[0].value.shape
                ) == 1 and in_nodes3[0].value.shape[0] == in_shape[-1]:
                    mode = "channel"
                    weight = self.op_mapper.weights[in_nodes3[0].layer_name]
                    weight = numpy.expand_dims(weight, 0)
                    weight = numpy.expand_dims(weight, 2)
                    weight = numpy.expand_dims(weight, 3)
                    self.op_mapper.weights[in_nodes3[0].layer_name] = weight
                    in_nodes3[0].fluid_code.layers[0].param_attr["shape"] = [
                        1, in_shape[-1], 1, 1
                    ]
                else:
                    is_prelu = False
                    continue

                if is_prelu:
                    index = in_nodes1.outputs.index(in_nodes0[0].layer_name)
                    del in_nodes1.outputs[index]
                    index = in_nodes1.outputs.index(in_nodes3[1].layer_name)
                    del in_nodes1.outputs[index]
                    index = in_nodes1.outputs.index(in_nodes4[1].layer_name)
                    del in_nodes1.outputs[index]
                    in_nodes1.outputs.append(node.layer_name)

                    node.layer_type = "Prelu"
                    node.inputs = [in_nodes1.layer_name]
                    act = node.fluid_code.layers[-1].param_attr.get("act", None)
                    node.fluid_code.clear()
                    attr = {
                        "mode": string(mode),
                        "param_attr": string(in_nodes3[0].layer_name)
                    }

                    node.fluid_code.add_layer(
                        "prelu",
                        inputs=in_nodes1.fluid_code.layers[-1].output,
                        output=node,
                        param_attr=attr)
                del self.graph.node_map[in_nodes0[0].layer_name]
                del self.graph.node_map[in_nodes0[1].layer_name]
                del self.graph.node_map[in_nodes2[0].layer_name]
                del self.graph.node_map[in_nodes2[1].layer_name]
                del self.graph.node_map[in_nodes3[1].layer_name]
                del self.graph.node_map[in_nodes4[1].layer_name]

    def merge_scale(self):
        for i, name in enumerate(self.graph.topo_sort):
            node = self.graph.get_node(name)
            if node is None:
                continue
            is_scale = True
            if node.layer_type == "Sub":
                in_nodes0 = [
                    self.graph.get_node(in_name) for in_name in node.inputs
                ]
                if in_nodes0[0].layer_type != "Mul" or in_nodes0[
                        1].layer_type != "Const" or in_nodes0[1].value.size != 1:
                    is_scale = False
                    continue
                if exist_act(in_nodes0[0]):
                    is_scale = False
                    continue
                if len(in_nodes0[0].outputs) != 1 or len(
                        in_nodes0[1].outputs) != 1:
                    is_scale = False
                    continue

                in_nodes1 = [
                    self.graph.get_node(in_name)
                    for in_name in in_nodes0[0].inputs
                ]
                if in_nodes1[0].layer_type != "Const" or in_nodes1[
                        1].layer_type != "RealDiv" or in_nodes1[
                            0].value.size != 1:
                    is_scale = False
                    continue
                if exist_act(in_nodes1[1]):
                    is_scale = False
                    continue
                if len(in_nodes1[0].outputs) != 1 or len(
                        in_nodes1[1].outputs) != 1:
                    is_scale = False
                    continue

                in_nodes2 = [
                    self.graph.get_node(in_name)
                    for in_name in in_nodes1[1].inputs
                ]
                if in_nodes2[1].layer_type != "Const" or in_nodes2[
                        1].value.size != 1:
                    is_scale = False
                    continue

                if is_scale:
                    in_node = self.graph.get_node(in_nodes1[1].inputs[0])
                    index = in_node.outputs.index(in_nodes1[1].layer_name)
                    in_node.outputs[index] = node.layer_name
                    node.layer_type = "Scale"
                    node.inputs = [in_node.layer_name]
                    scale = 1.0 / in_nodes2[1].value * in_nodes1[0].value
                    act = None
                    if node.fluid_code.layers[0].param_attr is not None:
                        act = node.fluid_code.layers[0].param_attr.get(
                            "act", None)
                    node.fluid_code.clear()

                    attr = {
                        "scale": scale,
                        "bias": in_nodes0[1].value,
                        "bias_after_scale": True,
                        "act": act
                    }
                    node.fluid_code.add_layer("scale",
                                              inputs=in_node,
                                              output=node,
                                              param_attr=attr)

                    del self.graph.node_map[in_nodes0[0].layer_name]
                    del self.graph.node_map[in_nodes0[1].layer_name]
                    del self.graph.node_map[in_nodes1[0].layer_name]
                    del self.graph.node_map[in_nodes1[1].layer_name]
                    del self.graph.node_map[in_nodes2[1].layer_name]

    def merge_affine_channel(self):
        for i, name in enumerate(self.graph.topo_sort):
            node = self.graph.get_node(name)
            if node is None:
                continue
            is_affine_channel = True
            if node.layer_type == "RealDiv":
                in_nodes0 = [
                    self.graph.get_node(in_name) for in_name in node.inputs
                ]
                bias_add = True
                if (in_nodes0[0].layer_type != "Sub" and in_nodes0[0].layer_type
                        != "Add") or in_nodes0[1].layer_type != "Const" or len(
                            in_nodes0[1].value.shape) != 3:
                    is_affine_channel = False
                    continue
                if in_nodes0[0].layer_type == "Sub":
                    bias_add = False
                if exist_act(in_nodes0[0]):
                    is_affine_channel = False
                    continue
                if len(in_nodes0[0].outputs) != 1 or len(
                        in_nodes0[1].outputs) != 1:
                    is_affine_channel = False
                    continue
                in_nodes1 = [
                    self.graph.get_node(in_name)
                    for in_name in in_nodes0[0].inputs
                ]
                if len(in_nodes1[0].out_shapes[0]
                       ) != 4 or in_nodes1[1].layer_type != "Const" or len(
                           in_nodes1[1].value.shape) != 3:
                    is_affine_channel = False
                    continue
                if len(in_nodes1[1].outputs) != 1:
                    is_affine_channel = False
                    continue
                channel = in_nodes1[0].out_shapes[0][-1]
                if channel < 0 or channel != in_nodes0[
                        1].value.size or channel != in_nodes1[1].value.size:
                    is_affine_channel = False
                    continue
                if in_nodes0[1].out_shapes[0][-1] != in_nodes0[
                        1].value.size or in_nodes1[1].out_shapes[0][
                            -1] != in_nodes1[1].value.size:
                    is_affine_channel = False
                    continue
                if is_affine_channel:
                    in_node = in_nodes1[0]
                    index = in_node.outputs.index(in_nodes0[0].layer_name)
                    in_node.outputs[index] = node.layer_name
                    node.layer_type = "AffineChannel"
                    node.inputs = [in_node.layer_name]
                    scale = 1.0 / in_nodes0[1].value.flatten()
                    bias = in_nodes1[1].value.flatten(
                    ) / in_nodes0[1].value.flatten()
                    if not bias_add:
                        bias *= -1.0
                    self.op_mapper.weights[node.layer_name + "_scale"] = scale
                    self.op_mapper.weights[node.layer_name + "_bias"] = bias

                    act = None
                    if node.fluid_code.layers[0].param_attr is not None:
                        act = node.fluid_code.layers[0].param_attr.get(
                            "act", None)
                    node.fluid_code.clear()

                    attr = {
                        "dtype": string(scale.dtype),
                        "shape": [channel],
                        "name": string(node.layer_name + "_scale")
                    }
                    node.fluid_code.add_layer("create_parameter",
                                              inputs=None,
                                              output=node.layer_name + "_scale",
                                              param_attr=attr)
                    attr = {
                        "dtype": string(scale.dtype),
                        "shape": [channel],
                        "name": string(node.layer_name + "_bias")
                    }
                    node.fluid_code.add_layer("create_parameter",
                                              inputs=None,
                                              output=node.layer_name + "_bias",
                                              param_attr=attr)
                    inputs = {
                        "x": in_node,
                        "scale": node.layer_name + "_scale",
                        "bias": node.layer_name + "_bias"
                    }
                    attr = {"act": act}
                    node.fluid_code.add_layer("affine_channel",
                                              inputs=inputs,
                                              output=node,
                                              param_attr=attr)

                    del self.graph.node_map[in_nodes0[0].layer_name]
                    del self.graph.node_map[in_nodes0[1].layer_name]
                    del self.graph.node_map[in_nodes1[1].layer_name]
