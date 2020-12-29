# -*- coding:UTF-8 -*-
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


import copy
import pandas as pd
from x2paddle.optimizer.pytorch_code_optimizer.layer_code_generator import rename_layers


def construct_attrs_table(sub_layers_list, node_name2sub_layers=None, module_name=None):
    """ 构造不同属性的表格。
    """
    def get_node_name(sub_layers):
        for k, v in node_name2sub_layers.items():
            if v == sub_layers:
                node_name = k
                break
        return node_name
    sub_layers = sub_layers_list[0]
    _, _, new_names = rename_layers(sub_layers)
    table = list()
    node_names = list()
    for i, sub_layers in enumerate(sub_layers_list):
        attrs = dict()
        if node_name2sub_layers is not None:
            node_names.append(get_node_name(sub_layers))
        else:
            node_names.append("{}_{}".format(module_name, i))
        for i, (layer_id, layer) in enumerate(sub_layers.items()):
            for k, v in layer.attrs.items():
                attrs[new_names[i] + "_{}".format(k)] = v
        table.append(attrs)
    pd_table = pd.DataFrame(table, index=node_names)
    return pd_table

def get_inputs_outputs(pd_graph, layers):
    inputs = list()
    outputs = list()
    cur_outputs = list()
    layer_ids = list(layers.keys())
    for layer_id, layer in layers.items():
        # 获取输出节点名字
        if layer_id not in pd_graph.edges_out:
            for index, output_name in enumerate(layer.outputs):
                if not output_name.startswith("x") or output_name in outputs \
                        or layer.kernel == "prim.assert":
                    continue
                elif layer.kernel == "prim.if" or layer.kernel == "prim.loop":
                        if index != 0:
                            outputs.append(output_name)
                elif output_name not in outputs:
                    outputs.append(output_name)
        else:
            for out_layer_id in pd_graph.edges_out[layer_id]:
                if out_layer_id not in layer_ids:
                    for index, output_name in enumerate(layer.outputs):
                        if not output_name.startswith("x") or output_name in outputs \
                                or layer.kernel == "prim.assert":
                            continue
                        elif layer.kernel == "prim.if" or layer.kernel == "prim.loop":
                            if index != 0:
                                outputs.append(output_name)
                        else:
                            outputs.append(output_name)
        # 获取输入节点名字
        for k, v in layer.inputs.items():
            if v not in cur_outputs and v not in inputs:
                inputs.append(v)
                
        if ("paddle.nn" in layer.kernel and "functional" not in layer.kernel):
            cur_outputs.extend(layer.outputs[1:])
        else:
            cur_outputs.extend(layer.outputs)
    return inputs, outputs

def get_inputs_count(pd_graph, sub_layers):
    input_ct2sub_layer_id = dict()
    for i, sub_layer in enumerate(sub_layers):
        inputs, outputs = get_inputs_outputs(pd_graph, sub_layer)
        if len(inputs) not in input_ct2sub_layer_id:
            input_ct2sub_layer_id[len(inputs)] = [i]
        else:
            input_ct2sub_layer_id[len(inputs)].append(i)
    return input_ct2sub_layer_id

def distinguish_sequential(pd_graph, module_name, sub_layers, sub_identifiers, node_name2sub_layers):
    """ 获取不同的layers组成的序列
    """
    def distinguish_sequential_by_inputs(part_layers, part_identifiers, part_module_name):
        new_sub_layers = dict()
        new_sub_sequentials = dict()
        sequentials2attrs_table = dict()
        input_ct2sub_layer_id = get_inputs_count(pd_graph, part_layers)
        if len(input_ct2sub_layer_id) == 1:
            new_sub_layers["{}".format(part_module_name)] = part_layers
            new_sub_sequentials["{}".format(part_module_name)] = part_identifiers
            sequentials2attrs_table["{}".format(part_module_name)] = construct_attrs_table(part_layers, node_name2sub_layers)
        else:
            for i, (k, indexes) in enumerate(input_ct2sub_layer_id.items()):
                new_sub_layers["{}__{}".format(part_module_name, i)] = list()
                new_sub_sequentials["{}__{}".format(part_module_name, i)] = list()
                for index in indexes:
                    new_sub_layers["{}__{}".format(part_module_name, i)].append(part_layers[index])
                    new_sub_sequentials["{}__{}".format(part_module_name, i)].append(part_identifiers[index])
                sequentials2attrs_table["{}__{}".format(part_module_name, i)] = \
                        construct_attrs_table(new_sub_layers["{}__{}".format(part_module_name, i)], node_name2sub_layers)
        return new_sub_layers, new_sub_sequentials, sequentials2attrs_table
        
    new_sub_layers = dict()
    new_sub_sequentials = dict()
    sequentials2attrs_table = dict()
    identifiers_str_list = list()
    for identifiers in sub_identifiers:
        identifiers_str_list.append(", ".join(list(identifiers.values())))
    identifiers_str_set = list(set(identifiers_str_list))
    if len(identifiers_str_set) == 1:
        return distinguish_sequential_by_inputs(sub_layers, sub_identifiers, module_name)
    else:
        for i in range(len(identifiers_str_set)):
            new_sub_layers["{}{}".format(module_name, i)] = list()
            new_sub_sequentials["{}{}".format(module_name, i)] = list()
    no_same_module_count = 0
    for j, identifiers in enumerate(sub_identifiers):
        identifiers_str = identifiers_str_list[j]
        for i in range(len(identifiers_str_set)):
            if identifiers_str_set[i] == identifiers_str:
                is_diff = False
                if identifiers_str_set[i].replace(", ", "").isdigit() or module_name == "ModuleList":
                    new_sub_layers["{}{}".format(module_name, len(identifiers_str_set) + no_same_module_count)] = [sub_layers[j]]
                    new_sub_sequentials["{}{}".format(module_name, len(identifiers_str_set) + no_same_module_count)] = [identifiers]
                    no_same_module_count += 1
                else:
                    new_sub_layers["{}{}".format(module_name, i)].append(sub_layers[j])
                    new_sub_sequentials["{}{}".format(module_name, i)].append(identifiers)
                break
    new_new_sub_layers = dict()
    new_new_sub_sequentials = dict()
    for k, v in new_sub_layers.items():
        part_sub_layers, part_sub_sequentials, part_sequentials2attrs_table = \
                    distinguish_sequential_by_inputs(v, new_sub_sequentials[k], k)
        new_new_sub_layers.update(part_sub_layers)
        new_new_sub_sequentials.update(part_sub_sequentials)
        sequentials2attrs_table.update(part_sequentials2attrs_table)
    return new_new_sub_layers, new_new_sub_sequentials, sequentials2attrs_table