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
import os.path as osp
from treelib import Tree
from queue import Queue
from x2paddle.optimizer.pytorch_code_optimizer.layer_code_generator import gen_layer_code, rename_layers, NN_KERNEL_WITH_PARAMS, NN_KERNEL_NAME
from x2paddle.optimizer.pytorch_code_optimizer.subgraphs_union import distinguish_sequential, get_inputs_outputs
from x2paddle.core.program import PaddleLayer
from x2paddle.optimizer.pytorch_code_optimizer.parameter_tree import PamareterNode, PamareterTree

SEPARATOR_IN_SCOPE = "/"


class HierarchicalTree(Tree):
    """ 定义层次树。
    """

    def __init__(self, pd_graph):
        super(HierarchicalTree, self).__init__()
        self.pd_graph = pd_graph
        self.script = pd_graph.script
        self.create_node("Module", self.pd_graph.name)  # create root
        self._hierarchical_order = dict()
        self.codes = list()
        self.identifier_idx = dict()
        self.param_tree = PamareterTree()
        self.module_name2count = dict()
        self.scope_name_list = list()

    def insert(self, layer):
        """ 往层次树中插入节点。

        Args:
            layer (PaddleLayer): 需要插入的节点。
        """
        scope_name = layer.scope_name
        self.scope_name_list.append(scope_name)
        if scope_name == "":
            if layer.kernel == "prim.tuple" or layer.kernel == "prim.tuple_unpack":
                layer_id = layer.id
                layer_id_list = list()
                for input_layer_id in self.pd_graph.edges_in[layer_id]:
                    layer_id_list.append(int(input_layer_id))
                layer_id_list = list(set(layer_id_list))
                layer_id_list.sort(reverse=True)

                if layer.kernel == "prim.tuple":
                    for i, input_layer_id in enumerate(layer_id_list):
                        input_layer_id_str = str(input_layer_id)
                        scope_name = self.pd_graph.layers[
                            input_layer_id_str].scope_name
                        if i == 0:
                            min_scope_name = scope_name
                        else:
                            len1 = len(min_scope_name.split("/"))
                            len2 = len(scope_name.split("/"))
                            if scope_name not in self.scope_name_list:
                                min_scope_name = scope_name
                                continue
                            if len1 > len2:
                                min_scope_name = scope_name
                    if min_scope_name == "":
                        self.create_node(
                            tag=layer.id,
                            identifier="no_scope_" + layer.id,
                            parent=self.pd_graph.name,
                            data=layer)
                        return
                    layer.scope_name = min_scope_name
                    scope_name = min_scope_name
                else:
                    for input_layer_id in layer_id_list:
                        input_layer_id_str = str(input_layer_id)
                        if self.pd_graph.layers[
                                input_layer_id_str].scope_name != "":
                            scope_name = self.pd_graph.layers[
                                input_layer_id_str].scope_name
                            break
                    layer.scope_name = scope_name
            else:
                self.create_node(
                    tag=layer.id,
                    identifier="no_scope_" + layer.id,
                    parent=self.pd_graph.name,
                    data=layer)
                return
        scopes = scope_name.split(SEPARATOR_IN_SCOPE)
        for idx, scope in enumerate(scopes):
            parent = SEPARATOR_IN_SCOPE.join(scopes[:idx])  #.lower()
            identifier = SEPARATOR_IN_SCOPE.join(scopes[:idx + 1])  #.lower()
            if self.contains(identifier):
                if idx != len(scopes) - 1:
                    parent_node = self.parent(identifier)
                    self.move_node(identifier, parent_node.identifier)
                    continue
                else:
                    if self.get_node(identifier).data is None:
                        data = layer
                        if identifier not in self.identifier_idx:
                            self.identifier_idx[identifier] = 0
                        else:
                            self.identifier_idx[identifier] += 1
                        identifier_name = identifier + SEPARATOR_IN_SCOPE + str(
                            self.identifier_idx[identifier])
                        self.create_node(
                            tag=scopes[idx],
                            identifier=identifier_name,
                            parent=identifier,
                            data=data)
                        data.scope_name = identifier_name
                        continue
                    else:
                        data = self[identifier].data
                        self[identifier].data = None
                        parent_node = self.parent(identifier)
                        self.move_node(identifier, parent_node.identifier)
                        if identifier not in self.identifier_idx:
                            self.identifier_idx[identifier] = 0
                        else:
                            self.identifier_idx[identifier] += 1
                        self.create_node(
                            tag=scopes[idx],
                            identifier=identifier + SEPARATOR_IN_SCOPE +
                            str(self.identifier_idx[identifier]),
                            parent=identifier,
                            data=data)
                        self.identifier_idx[identifier] += 1
                        data = layer
                        self.create_node(
                            tag=scopes[idx],
                            identifier=identifier + SEPARATOR_IN_SCOPE +
                            str(self.identifier_idx[identifier]),
                            parent=identifier,
                            data=data)
                        continue
            if idx == 0 and not self.contains(identifier):
                data = layer if idx == len(scopes) - 1 else None
                self.create_node(
                    tag=scopes[idx],
                    identifier=identifier,
                    parent=self.pd_graph.name,
                    data=data)
            else:
                if idx == len(scopes) - 1:
                    if parent == "":
                        childs = self.children(self.pd_graph.name)
                        parent = self.pd_graph.name
                    else:
                        childs = self.children(parent)
                    prefix = identifier
                    identifiers = list()
                    for child in childs:
                        child_identifier = child.identifier
                        if child_identifier.startswith(
                                prefix) and child_identifier != prefix:
                            identifiers.append(child_identifier)
                    if len(identifiers) == 0:
                        identifier = prefix + "_0"
                    else:
                        identifier_ids = list()
                        for id_obj in identifiers:
                            identifier_ids.append(int(id_obj.split("_")[-1]))
                        identifier_ids.sort()
                        identifier = prefix + "_{}".format(identifier_ids[-1] +
                                                           1)
                data = layer if idx == len(scopes) - 1 else None
                self.create_node(
                    tag=scopes[idx],
                    identifier=identifier,
                    parent=parent,
                    data=data)

    def update_hierarchical_order(self):
        """ 更新层次排序，使用一个字典存储该信息，
            关键字为当前层次，值为节点名字。
        """
        hierarchical_order = dict()
        queue = Queue()
        queue.put(item=(self.pd_graph.name, 0), block=False)
        while not queue.empty():
            node_name, cur_level = queue.get(block=False)
            node_inst = self[node_name]
            if cur_level not in hierarchical_order:
                hierarchical_order[cur_level] = []
            hierarchical_order[cur_level].append(node_name)
            for successor_name in node_inst.successors(self.identifier):
                queue.put(item=(successor_name, cur_level + 1), block=False)
        self._hierarchical_order = hierarchical_order

    def analyze_attrs_table(self, attrs_table):
        """ 分析属性表格，哪些属性取值不一致。
        """
        diff_attrs_column = list()
        for column in list(attrs_table.columns):
            elements = list(attrs_table.get(column))
            base = elements[0]
            for element in elements[1:]:
                if isinstance(base, str) and "'" not in base:
                    break
                if element != base:
                    diff_attrs_column.append(column)
                    break
        return diff_attrs_column

    def merge_node(self, sub_layers_list, attrs_table, node_name2sub_layers,
                   module_name):
        """ 将一个scope的节点合成一个Module（Class），并将对应的Class代码
            放到code字符串中。
        """

        def get_node_name(sub_layers):
            for k, v in node_name2sub_layers.items():
                if v == sub_layers:
                    node_name = k
                    break
            return node_name

        sub_layers = sub_layers_list[0]
        node_name = get_node_name(sub_layers)

        sub_layers, _, _ = rename_layers(sub_layers)
        diff_attrs_column = self.analyze_attrs_table(attrs_table)
        if module_name is None:
            module_name = node_name.replace("/", "_")  #node_name.split("/")[-1]
            module_name = module_name[0].upper() + module_name[1:]
        if module_name in self.module_name2count:
            module_name = module_name + "_0"
        code_str = gen_layer_code(
            self.pd_graph,
            sub_layers,
            module_name,
            different_attrs=diff_attrs_column)

        self.codes.append(code_str)
        for sub_layers in sub_layers_list:
            inputs, outputs = get_inputs_outputs(self.pd_graph, sub_layers)
            inputs_dict = dict()
            for i, input in enumerate(inputs):
                inputs_dict["input_{}".format(i)] = input
            if module_name in self.module_name2count:
                self.module_name2count[module_name] += 1
            else:
                self.module_name2count[module_name] = 0
            if module_name.lower() in NN_KERNEL_NAME.values():
                mn = module_name.lower() + "__"
            else:
                mn = module_name.lower()
            outputs = [
                "{}/{}".format(mn, self.module_name2count[module_name])
            ] + outputs
            node_name = get_node_name(sub_layers)
            diff_attrs = dict()
            for column in diff_attrs_column:
                diff_attrs[column] = attrs_table.get(column).loc[node_name]

            node_name_seg = node_name.split(SEPARATOR_IN_SCOPE)
            node_name_seg[-1] = module_name.lower()
            new_node_name = SEPARATOR_IN_SCOPE.join(node_name_seg)
            new_layer = PaddleLayer(
                id=list(sub_layers.keys())[-1],
                kernel="module",
                inputs=inputs_dict,
                outputs=outputs,
                scope_name=new_node_name,
                module=module_name,
                **diff_attrs)

            _, nn_param_nodes, _ = rename_layers(sub_layers, self.param_tree)
            param_node = PamareterNode(old_name=outputs[0])
            for node in nn_param_nodes:
                param_node.add_child(node)
            self.param_tree.add_node(param_node)

            for i, (layer_id, layer) in enumerate(sub_layers.items()):
                if i == len(sub_layers) - 1:
                    self.pd_graph.layers[layer_id] = new_layer
                else:
                    self.pd_graph.layers.pop(layer_id)

            self.pd_graph.build()
            self[node_name].data = new_layer

    def find_subgraph_diff(self, module_name2sub_layers,
                           module_name2sub_identifiers, node_name2sub_layers,
                           name):
        """ 查找子图的diff，主要是输入参数的diff。
        """
        sub_layers = module_name2sub_layers[name]
        sub_identifiers = module_name2sub_identifiers[name]
        new_sub_layers, new_sub_sequentials, sequentials2attrs_table = distinguish_sequential(
            self.pd_graph, name, sub_layers, sub_identifiers,
            node_name2sub_layers)
        module_name2sub_layers.pop(name)
        module_name2sub_identifiers.pop(name)
        for k, v in new_sub_layers.items():
            module_name2sub_layers[k] = v
            module_name2sub_identifiers[k] = new_sub_sequentials[k]
        return sequentials2attrs_table

    def convert_subgraph_to_layer(self):
        """
            1. 根据_hierarchical_order，从最深的层次开始将
               子图合并成layer（即合成节点）。
            2. 根据参数名新旧对应关系，更新参数名。
        """
        depths = sorted(list(self._hierarchical_order.keys()), reverse=True)
        all_name_old2new = dict()
        current_module_name_list = list()
        for depth in depths[1:]:
            # Module的名字与子图的对应关系
            module_name2sub_layers = dict()
            # Module的名字与子图中layer命名的对应关系
            module_name2sub_identifiers = dict()
            # 层次树中包含子树的节点，其节点名与子图对用关系
            node_name2sub_layers = dict()
            for node_name in self._hierarchical_order[depth]:
                node_inst = self[node_name]
                if node_inst.data is None:
                    sub_layers = dict()
                    sub_identifiers = dict()
                    for successor_name in node_inst.successors(self.identifier):
                        sub_layers[self[successor_name].data.id] = self[
                            successor_name].data
                        sub_identifiers[self[successor_name].data.id] = self[
                            successor_name].data.scope_name.split("/")[-1]
                    node_name2sub_layers[node_name] = sub_layers
                    node_name_segs = node_name.split("/")

                    # 获取Module的名字
                    module = self.script
                    is_largest_module = False  # 当前module是否是最外层的Module
                    for name_id, name in enumerate(node_name_segs):
                        name = name.split("__")[0]
                        if not hasattr(module, name):
                            is_largest_module = True
                            break
                        module = getattr(module, name)
                    if is_largest_module:
                        if name_id == 0:
                            module_name = name
                        else:
                            module_name = "_".join(node_name_segs)
                    else:
                        module_name = module._get_name()
                    if module_name in module_name2sub_layers:
                        if len(sub_layers[list(sub_layers.keys())[-1]].outputs) != \
                                    len(module_name2sub_layers[module_name][0][list(module_name2sub_layers[module_name][0].keys())[-1]].outputs):
                            while module_name in module_name2sub_layers:
                                module_name = module_name + "__tmp"
                                if module_name in module_name2sub_layers and \
                                        len(sub_layers[list(sub_layers.keys())[-1]].outputs) == \
                                        len(module_name2sub_layers[module_name][0][list(module_name2sub_layers[module_name][0].keys())[-1]].outputs):
                                    break
                            if module_name not in module_name2sub_layers:
                                module_name2sub_layers[
                                    module_name] = [sub_layers]
                                module_name2sub_identifiers[
                                    module_name] = [sub_identifiers]
                            else:
                                module_name2sub_layers[module_name].append(
                                    sub_layers)
                                module_name2sub_identifiers[module_name].append(
                                    sub_identifiers)
                        else:
                            module_name2sub_layers[module_name].append(
                                sub_layers)
                            module_name2sub_identifiers[module_name].append(
                                sub_identifiers)
                    else:
                        module_name2sub_layers[module_name] = [sub_layers]
                        module_name2sub_identifiers[
                            module_name] = [sub_identifiers]
            module_names = list(module_name2sub_layers.keys())
            for module_name in module_names:
                sequentials2attrs_table = self.find_subgraph_diff(
                    module_name2sub_layers, module_name2sub_identifiers,
                    node_name2sub_layers, module_name)
                for name in sequentials2attrs_table.keys():
                    if name.startswith("Sequential"):
                        # 若Module的名字为Sequential，则以scope_name的名字来命名，在merge_node中实现
                        module_name = None
                    else:
                        module_name = name
                        while module_name in current_module_name_list:
                            module_name += "__0"
                    current_module_name_list.append(module_name)
                    self.merge_node(module_name2sub_layers[name],
                                    sequentials2attrs_table[name],
                                    node_name2sub_layers, module_name)

    def update_parameters(self):
        """ 更新参数。
        """
        self.param_tree.traverse()
        full_old_name_list = copy.deepcopy(
            list(self.pd_graph.parameters.keys()))
        for old_name, new_name in self.param_tree.old2new.items():
            for full_old_name in full_old_name_list:
                if full_old_name.startswith("{}.".format(old_name)):
                    full_new_name = full_old_name.replace(
                        "{}.".format(old_name), "{}.".format(new_name))
                    params = self.pd_graph.parameters.pop(full_old_name)
                    self.pd_graph.parameters[full_new_name] = params
                if full_old_name == old_name:
                    full_new_name = full_old_name.replace(old_name, new_name)
                    params = self.pd_graph.parameters.pop(full_old_name)
                    self.pd_graph.parameters[full_new_name] = params

    def save_source_files(self, save_dir):
        def gen_main_code():
            input_data_name = ', '.join(self.pd_graph.inputs)
            run_func_list = list()
            run_func_list.append("def main({}):".format(input_data_name))
            run_func_list.append("    # There are {} inputs.".format(
                len(self.pd_graph.inputs_info)))
            for k, v in self.pd_graph.inputs_info.items():
                run_func_list.append("    # {}: shape-{}, type-{}.".format(k, v[
                    0], v[1]))
            run_func_list.extend([
                "    paddle.disable_static()",
                "    params = paddle.load('{}')".format(
                    osp.join(osp.abspath(save_dir), "model.pdparams")),
                "    model = {}()".format(self.pd_graph.name),
                "    model.set_dict(params)", "    model.eval()",
                "    out = model({})".format(input_data_name), "    return out"
            ])
            return "\n".join(run_func_list)

        self.update_hierarchical_order()
        self.convert_subgraph_to_layer()
        self.update_parameters()
        import_list = ["import paddle",
                       "import math",
                       "from x2paddle.op_mapper.pytorch2paddle " + \
                                 "import pytorch_custom_layer as x2paddle_nn",
                       "",]
        import_str = "\n".join(import_list) + "\n"
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        f = open(osp.join(save_dir, 'x2paddle_code.py'), 'w')
        f.write(import_str)
        for code in self.codes:
            f.write(code)
            f.write("\n")
        run_func = gen_main_code()
        f.write(run_func)
        f.close()
