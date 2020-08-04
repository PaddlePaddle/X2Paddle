#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import torch
from x2paddle.core.util import *


def prim_GetAttr(mapper, graph, node):
    node_name = mapper._get_node_name(node)[0]
    field_name_list = [node.s('name')]
    while True:
        input_node = list(node.inputs())[0].node()
        try:
            field_name_list.insert(0, input_node.s('name'))
            node = input_node
        except Exception:
            break
    part_script = mapper.script
    for field_name in field_name_list:
        if hasattr(part_script, field_name):
            param = getattr(part_script, field_name)
            if isinstance(param, torch.Tensor):
                param = param.detach().numpy()
            mapper.pytorch_params[node_name] = param
            part_script = param
    return [node_name], []


def prim_Constant(mapper, graph, node):
    node_name = mapper._get_node_name(node)[0]
    output = list(node.outputs())[0]
    value = output.toIValue()
    mapper.attrs[node_name] = value
    if isinstance(value, str):
        value = string(value)
    graph.add_layer(
        "prim.constant", inputs={}, outputs=[node_name], value=value)
    return [node_name], []


def prim_ListConstruct(mapper, graph, node):
    node_name = mapper._get_node_name(node)[0]
    inputs = {}
    for i, input_ivalue in enumerate(node.inputs()):
        input_node = input_ivalue.node()
        input_unique_id = input_ivalue.unique()
        input_node_name = mapper.node_names[input_unique_id]
        inputs['input{}'.format(i)] = input_node_name
    graph.add_layer("prim.list", inputs=inputs, outputs=[node_name])
    return [node_name], list(inputs.values())


def prim_RaiseException(mapper, graph, node):
    node_name = mapper._get_node_name(node)[0]
    node_name_list = [node_name]
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.node_names[input_unique_id]
    mapper._check_input(graph, input_node, input_node_name, node_name_list)
    graph.add_layer(
        "prim.exception",
        inputs={'input': input_node_name},
        outputs=[node_name])
    return node_name_list, [input_node_name]


def prim_Loop(mapper, graph, node):
    node_name = mapper._get_node_name(node)[0]
    node_name_list = [node_name]
    loop_inputs = {}
    block = list(node.blocks())[0]
    loop_outputs = [node_name]
    for i, block_input_ivalue in enumerate(block.inputs()):
        block_input_node_name = 'x' + str(mapper.node_index)
        unique_id = block_input_ivalue.unique()
        if unique_id not in mapper.node_names:
            mapper.node_names[unique_id] = block_input_node_name
            mapper.node_index += 1
        if i == 0:
            loop_input_node = list(node.inputs())[0].node()
            loop_input_unique_id = list(node.inputs())[0].unique()
            loop_input_node_name = mapper.node_names[loop_input_unique_id]
            mapper._check_input(graph, loop_input_node, loop_input_node_name,
                                node_name_list)
            loop_inputs['input'] = loop_input_node_name
            loop_outputs.append(block_input_node_name)
            node_name_list.append(block_input_node_name)
        else:
            loop_input_node = list(node.inputs())[i + 1].node()
            loop_input_unique_id = list(node.inputs())[i + 1].unique()
            loop_input_node_name = mapper.node_names[loop_input_unique_id]
            mapper._check_input(graph, loop_input_node, loop_input_node_name,
                                node_name_list)
            graph.add_layer(
                "prim.equal",
                inputs={'input': loop_input_node_name},
                outputs=[block_input_node_name])
            node_name_list.append(block_input_node_name)
    graph.add_layer("prim.loop", inputs=loop_inputs, outputs=loop_outputs)
    current_layer = graph.layers[-1]
    block_graph, graph_inputs = mapper.traverse(block, node)
    for i, input_name in enumerate(graph_inputs):
        if input_name == loop_outputs[1]:
            continue
        current_layer.inputs['input-{}'.format(i)] = input_name
    current_layer.add_block(block_graph)
    return node_name_list, list(current_layer.inputs.values())


def prim_If(mapper, graph, node):
    node_name = mapper._get_node_name(node)[0]
    node_name_list = [node_name]
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.node_names[input_unique_id]
    mapper._check_input(graph, input_node, input_node_name, node_name_list)
    graph.add_layer("prim.if", {'input': input_node_name}, [node_name])
    current_layer = graph.layers[-1]
    block0 = list(node.blocks())[0]
    block0_graph, graph_inputs0 = mapper.traverse(block0, node)
    len0 = 0
    for i, input_name in enumerate(graph_inputs0):
        current_layer.inputs['input-{}'.format(i)] = input_name
        len0 = i
    current_layer.add_block(block0_graph)
    block1 = list(node.blocks())[1]
    block1_graph, graph_inputs1 = mapper.traverse(block1, node)
    for i, input_name in enumerate(graph_inputs1):
        current_layer.inputs['input-{}'.format(len0 + 1 + i)] = input_name
    current_layer.add_block(block1_graph)
    return node_name_list, list(current_layer.inputs.values())


def prim_min(mapper, graph, node):
    node_name = mapper._get_node_name(node)[0]
    node_name_list = [node_name]
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.node_names[input_unique_id]
    mapper._check_input(graph, input_node, input_node_name, node_name_list)
    graph.add_layer(
        "prim.min", inputs={'input': input_node_name}, outputs=[node_name])
    return node_name_list, [input_node_name]
