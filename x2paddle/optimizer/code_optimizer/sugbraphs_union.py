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


import pandas as pd
from x2paddle.optimizer.code_optimizer.layer_code_generator import rename_layers


def construct_attrs_table(sub_layers_list, node_name2sub_layers):
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
    for sub_layers in sub_layers_list:
        attrs = dict()
        node_names.append(get_node_name(sub_layers))
        for i, (layer_id, layer) in enumerate(sub_layers.items()):
            for k, v in layer.attrs.items():
                attrs[new_names[i] + "_{}".format(k)] = v
        table.append(attrs)
    pd_table = pd.DataFrame(table, index=node_names)
    return pd_table

def distinguish_sequential(module_name, sub_layers, sub_identifiers, identifiers_list, node_name2sub_layers):
    new_sub_layers = dict()
    new_sub_sequentials = dict()
    sequentials2attrs_table = dict()
    identifiers_str_list = list()
    for identifiers in identifiers_list:
        identifiers_str_list.append(", ".join(identifiers))
    identifiers_str_list = list(set(identifiers_str_list))
    if len(identifiers_str_list) == 1:
        new_sub_layers["{}".format(module_name)] = sub_layers
        new_sub_sequentials["{}".format(module_name)] = sub_identifiers
        sequentials2attrs_table["{}".format(module_name)] = construct_attrs_table(sub_layers, node_name2sub_layers)
        return new_sub_layers, new_sub_sequentials, sequentials2attrs_table
    else:
        for i in range(len(identifiers_str_list)):
            new_sub_layers["{}{}".format(module_name, i)] = list()
            new_sub_sequentials["{}{}".format(module_name, i)] = list()
    for j, identifiers in enumerate(sub_identifiers):
        identifiers_str = ", ".join(identifiers_list[j])
        for i in range(len(identifiers_str_list)):
            if identifiers_str_list[i] == identifiers_str:
                new_sub_layers["{}{}".format(module_name, i)].append(sub_layers[j])
                new_sub_sequentials["{}{}".format(module_name, i)].append(identifiers)
                continue
    for k, v in new_sub_layers.items():
        sequentials2attrs_table[k] = construct_attrs_table(v, node_name2sub_layers)
    return new_sub_layers, new_sub_sequentials, sequentials2attrs_table