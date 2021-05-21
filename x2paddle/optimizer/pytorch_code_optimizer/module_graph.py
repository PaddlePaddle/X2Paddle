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
import os
import os.path as osp
from x2paddle.core.program import PaddleLayer
from x2paddle.optimizer.pytorch_code_optimizer.subgraphs_union import construct_attrs_table, get_inputs_outputs
from x2paddle.optimizer.pytorch_code_optimizer.layer_code_generator import gen_layer_code, rename_layers
from x2paddle.optimizer.pytorch_code_optimizer.parameter_tree import PamareterNode, PamareterTree


NoModuleStart = ["paddle.nn.ReLU"]

class Apriori(object):
    """ 使用Apriori算法挖掘频繁子图
    1. 构建频繁1项集
    2. 挖掘频繁k项集
    3. 最终k项集和节点数满足最少节点数的子图组成集合GS
    
    Args:
       min_support (int): 子图出现次数的最小值。
    """
    def __init__(self, min_support):
        self.min_support = min_support 
        
    def is_match(self, item, sublayers):
        for i in range(len(item)):
            if len(sublayers) <= i or item[i] != sublayers[i].kernel:
                return False
        return True
    
    def create_C1(self):
        # 构建候选1-项集
        C1 = list()
        for layer_id, layer in self.layers.items():
            if layer.kernel == "paddle.to_tensor" or \
                    layer.kernel == "prim.if" or \
                    layer.kernel == "prim.loop": #or \
#                     layer.kernel == "prim.list" or \
#                     layer.kernel == "prim.tuple" or \
#                     layer.kernel == "prim.dict_construct":
                continue
            if self.pd_graph.edges_in.get(layer_id, 0) == 0 and \
                    self.pd_graph.edges_out.get(layer_id, 0) == 0:
                continue
            if [layer.kernel] not in C1:
                C1.append([layer.kernel])
        return C1
        
    def create_Ck(self, Lk_last, C1):
        # 构建候选k-项集
        Ck = list()
        for item in Lk_last:
            for item_one in C1:
                new_item = copy.deepcopy(item)
                new_item.append(item_one[0])
                if new_item[0] in NoModuleStart:
                    continue
                Ck.append(new_item)
        return Ck
        
    def generate_Lk_by_Ck(self, Ck):
        # 生成频繁k-项集
        Lk = list()
        for item in Ck:
            count = 0
            for i in range(len(self.layers)):
                sublayers = list(self.layers.values())[i:]
                if self.is_match(item, sublayers):
                    count += 1
            if count >= self.min_support:
                Lk.append(item) 
        return Lk
                
    def run(self, graph):
        self.pd_graph = graph
        self.layers = graph.layers
        itemset = list()
        C1 = self.create_C1()
        L1 = self.generate_Lk_by_Ck(C1)
        Lk = L1
        while len(Lk) > 0:
            Ck = self.create_Ck(Lk, C1)
            Lk = self.generate_Lk_by_Ck(Ck)
            itemset.extend(Lk)
        return itemset
        

class DP(object):
    """ 使用动动态规划找到使代码最短的组合方式。
    """
    def __init__(self, combination_itemset):
        self.combination_itemset = combination_itemset
        
    def get_combination_id(self, combination, layers):
        combination_id = list()
        for layer_obj in combination:
            if len(layer_obj) > 1:
                kernel_itemset = list()
                for layer_id in layer_obj:
                    kernel_itemset.append(layers[layer_id].kernel)
                id = self.combination_itemset.index(kernel_itemset)
                combination_id.append(id)
            else:
                combination_id.append(-1)
        return combination_id
        
    def run(self, graph):
        layers = graph.layers
        layer_combination_list = list()
        for i, (layer_id, layer) in enumerate(layers.items()):
            if i == 0:
                layer_combination_list.append([[layer_id]])
                continue
            current_itemset = [layer_id]
            kernel_itemset = [layer.kernel]
            candidate_itemset = list()
            min_count = len(layers)
            prefix_ids = list(range(i))
            prefix_ids.reverse()
            for j in prefix_ids:
                current_layer_id = list(layers.keys())[j]
                current_layer = list(layers.values())[j]
                current_itemset.insert(0, current_layer_id)
                kernel_itemset.insert(0, current_layer.kernel) 
                if kernel_itemset in self.combination_itemset:
                    current_count = len(layer_combination_list[j - 1])
                    all_count = current_count + 1
                    if all_count < min_count:
                        min_count = all_count
                        candidate_itemset = copy.deepcopy(current_itemset)
                        if j - 1 < 0:
                            last_itemset = list()
                        else:
                            last_itemset = copy.deepcopy(layer_combination_list[j - 1])
                else:
                    if j == prefix_ids[0]:
                        min_count = len(layer_combination_list[j]) + 1
                        current_itemset.pop(0)
                        candidate_itemset = copy.deepcopy(current_itemset)
                        last_itemset = copy.deepcopy(layer_combination_list[j])
                    break
            last_itemset.append(candidate_itemset)
            layer_combination_list.append(last_itemset)
        final_combination = layer_combination_list[-1]
        combination_id = self.get_combination_id(final_combination, layers)
        return final_combination, combination_id
                    
        
class ModuleGraph(object):
    """ 更新PaddleGraph，生成代码。
    """
    def __init__(self, graph):
        self.pd_graph = graph
        self.global_layers = graph.get_global_layers()
        self.codes = list()
        self.param_tree = PamareterTree()
    
    def get_updation_information(self):
        aprior = Apriori(3)
        combination_itemset = aprior.run(self.pd_graph)
        dp = DP(combination_itemset)
        combination, combination_id = dp.run(self.pd_graph)
        return combination, combination_id
    
    def analyze_attrs_table(self, attrs_table):
        """ 分析属性表格，哪些属性取值不一致。
        """
        diff_attrs_column = dict()
        for column in list(attrs_table.columns):
            elements = list(attrs_table.get(column))
            elements_list = list()
            count_list = list()
            for element in elements:
                if isinstance(element, str) and "'" not in element:
                    break
                if element not in elements_list:
                    count_list.append(1)
                    elements_list.append(element)
                else:
                    index = elements_list.index(element)
                    count_list[index] += 1
            if len(elements_list) > 1:
                max_ct = 0
                for k, v in zip(elements_list, count_list):
                    if v > max_ct and str(k) != "nan" :
                        max_ele = k
                        max_ct = v
                diff_attrs_column[column] = max_ele
        return diff_attrs_column
    
    def analyze_graph(self, sub_layers_list):
        def is_same(sub_layers1, sub_layers2, id1, id2):
            inputs1, outputs1 = ipt_opt_list[id1]
            inputs2, outputs2 = ipt_opt_list[id2]
            if len(inputs1) != len(inputs2) or len(outputs1) != len(outputs2):
                return False
            layer_id_list1 = list(sub_layers1.keys())
            layer_id_list2 = list(sub_layers2.keys())
            for i, layer_id1 in enumerate(layer_id_list1):
                layer_id2 = layer_id_list2[i]
                if layer_id2 not in self.pd_graph.edges_in:
                    return False
                if len(self.pd_graph.edges_in[layer_id1]) != len(self.pd_graph.edges_in[layer_id2]):
                    return False
                for j, ipt_layer_id1 in enumerate(self.pd_graph.edges_in[layer_id1]):
                    ipt_layer_id2 = self.pd_graph.edges_in[layer_id2][j]
                    if (ipt_layer_id1 in layer_id_list1) ^ (ipt_layer_id2 in layer_id_list2):
                        return False
                if (layer_id1 in self.pd_graph.edges_out) ^ (layer_id2 in self.pd_graph.edges_out):
                    return False
                if (layer_id1 in self.pd_graph.edges_out) and (layer_id2 in self.pd_graph.edges_out):
                    if (len(self.pd_graph.edges_out[layer_id1]) > 1 and len(self.pd_graph.edges_out[layer_id2]) == 1) or \
                       (len(self.pd_graph.edges_out[layer_id1]) == 1 and len(self.pd_graph.edges_out[layer_id2]) > 1):
                        return False
                    for j, opt_layer_id1 in enumerate(self.pd_graph.edges_out[layer_id1]):
                        if len(self.pd_graph.edges_out[layer_id1]) == 1 and len(self.pd_graph.edges_out[layer_id2]) == 1:
                            opt_layer_id2 = self.pd_graph.edges_out[layer_id2][j]
                            if (opt_layer_id1 in layer_id_list1) ^ (opt_layer_id2 in layer_id_list2):
                                return False
            return True
        sub_layers_list_list = list()
        id_list = list()
        ipt_opt_list = list()
        sub_layers_list_list.append([sub_layers_list[0]])
        id_list.append(0)
        for i, sub_layer in enumerate(sub_layers_list):
            ipt_opt_list.append(get_inputs_outputs(self.pd_graph, sub_layer))
            if i == 0:
                continue
            id_list_cp = copy.deepcopy(id_list)
            for j, index in enumerate(id_list_cp):
                if is_same(sub_layers_list[index], sub_layer, index, i):
                    sub_layers_list_list[j].append(sub_layer)
                    break
                if j == len(id_list_cp) - 1:
                    sub_layers_list_list.append(list())
                    sub_layers_list_list[j + 1].append(sub_layer)
                    id_list.append(i)
        return sub_layers_list_list
                
    
    def merge_node(self, sub_layers_list, attrs_table, module_name):
        sub_layers = sub_layers_list[0]
        diff_attrs_column = self.analyze_attrs_table(attrs_table)
        sub_layers, _, _ = rename_layers(sub_layers)
        code_str = gen_layer_code(self.pd_graph, 
                                  sub_layers, 
                                  module_name, 
                                  different_attrs=diff_attrs_column)
        self.codes.append(code_str)
        for index, sub_layers in enumerate(sub_layers_list):
            inputs, outputs = get_inputs_outputs(self.pd_graph, sub_layers)
            inputs_dict = dict()
            for i, input in enumerate(inputs):
                inputs_dict["input_{}".format(i)] = input
            mn = module_name.lower()
            outputs = ["{}_{}".format(mn, index)] + outputs
            node_name = "{}_{}".format(module_name, index)
            diff_attrs = dict() 
            for column, element in diff_attrs_column.items():
                current_element = attrs_table.get(column).loc[node_name]
                if current_element != element:
                    diff_attrs[column] = current_element
            new_layer = PaddleLayer(id=list(sub_layers.keys())[-1],
                                    kernel="module",
                                    inputs=inputs_dict,
                                    outputs=outputs,
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
                    if len(layer_id.split(".")) > 1:
                        continue
                    self.pd_graph.layers.pop(layer_id)

            self.pd_graph.build()
    
    def convert_subgraph_to_layer(self, combination, combination_id):
        combination_id_set = set(combination_id)
        for s in list(combination_id_set):
            if s == -1:
                continue
            module_name = "Block{}".format(s)
            sub_layers_list = list()
            for i, c in enumerate(combination):
                if len(c) > 1 and combination_id[i] == s:
                    sub_layers = dict()
                    for layer_id in c:
                        sub_layers[layer_id] = self.global_layers[layer_id]
                    sub_layers_list.append(sub_layers)
            sub_layers_list_list = self.analyze_graph(sub_layers_list)
            for i, sub_layers_list in enumerate(sub_layers_list_list):
                if i == 0:
                    real_module_name = module_name
                else:
                    real_module_name = module_name + "__{}".format(i)
                if len(sub_layers_list) > 1:
                    attrs_table = construct_attrs_table(sub_layers_list, module_name=real_module_name)
                    self.merge_node(sub_layers_list, attrs_table, real_module_name)
        layers, nn_param_nodes, _ = rename_layers(self.pd_graph.layers, self.param_tree, is_rename_module=True)
        code_str = gen_layer_code(self.pd_graph, 
                                  layers, 
                                  self.pd_graph.name)
        self.codes.append(code_str)
        param_node = PamareterNode(old_name="Module")
        for node in nn_param_nodes:
            param_node.add_child(node)
        self.param_tree.add_node(param_node)
        
    def update_parameters(self):
        """ 更新参数。
        """
        self.param_tree.traverse()
        full_old_name_list = copy.deepcopy(list(self.pd_graph.parameters.keys()))
        for old_name, new_name in self.param_tree.old2new.items():
            for full_old_name in full_old_name_list:
                if full_old_name.startswith("{}.".format(old_name)):
                    full_new_name = full_old_name.replace("{}.".format(old_name), "{}.".format(new_name))
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
            run_func_list.append("    # There are {} inputs.".format(len(self.pd_graph.inputs_info)))
            for k, v in self.pd_graph.inputs_info.items():
                run_func_list.append("    # {}: shape-{}, type-{}.".format(k, v[0], v[1]))
            run_func_list.extend(
                ["    paddle.disable_static()",
                 "    params = paddle.load('{}')".format(osp.join(osp.abspath(save_dir), "model.pdparams")),
                 "    model = {}()".format(self.pd_graph.name),
                 "    model.set_dict(params)",
                 "    model.eval()",
                 "    out = model({})".format(input_data_name),
                 "    return out"])
            return "\n".join(run_func_list)
        combination, combination_id = self.get_updation_information()
        self.convert_subgraph_to_layer(combination, combination_id)
        self.update_parameters()
        import_list = ["import paddle",
                       "import math",
                       "from x2paddle.op_mapper.pytorch2paddle " + \
                                 "import pytorch_custom_layer as x2paddle_nn"
                       "\n",]
        import_str = "\n".join(import_list)
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
    