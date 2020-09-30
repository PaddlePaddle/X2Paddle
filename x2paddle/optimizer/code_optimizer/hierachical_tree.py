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


import os.path as osp
from treelib import Tree
from queue import Queue
from x2paddle.optimizer.code_optimizer.layer_code_generator import gen_layer_code, rename_layers, NN_KERNEL_WITH_PARAMS
from x2paddle.core.program import PaddleLayer

SEPARATOR_IN_SCOPE = "/"


class HierarchicalTree(Tree):
    """ 定义层次树。
    """
    def __init__(self, pd_graph):
        super(HierarchicalTree, self).__init__()
        self.pd_graph = pd_graph
        self.create_node("Module", self.pd_graph.name) # create root
        self._hierarchical_order = dict()
        self.codes = list()
        
    def insert(self, layer):
        """ 往层次树中插入节点。
        
        Args:
            layer (PaddleLayer): 需要插入的节点。
        """
        scope_name = layer.scope_name
        if scope_name == "":
            self.create_node(tag=layer.id, 
                             identifier="no_scope_" + layer.id, 
                             parent=self.pd_graph.name,
                             data=layer)
            return 
        scopes = scope_name.split(SEPARATOR_IN_SCOPE)
        for idx, scope in enumerate(scopes):
            parent = SEPARATOR_IN_SCOPE.join(scopes[:idx])
            identifier = SEPARATOR_IN_SCOPE.join(scopes[:idx + 1])
            if self.contains(identifier) and idx != len(scopes) - 1:
                cur_node = self[identifier]
                parent_node = self.parent(identifier)
                self.move_node(identifier, parent_node.identifier)
                continue
            if idx == 0:
                data = layer if idx == len(scopes) - 1 else None
                self.create_node(tag=scopes[idx], 
                                 identifier=identifier.lower(), 
                                 parent=self.pd_graph.name,
                                 data=data)
            else:
                if idx == len(scopes) - 1:
                    childs = self.children(parent)
                    prefix = identifier.lower()
                    identifiers = list()
                    for child in childs:
                        child_identifier = child.identifier
                        if child_identifier.startswith(prefix):
                            identifiers.append(child_identifier)
                    if len(identifiers) == 0:
                        identifier = prefix + "_0"
                    else:
                        identifiers.sort()
                        last_idx = int(identifiers[-1].split("_")[-1])
                        identifier = prefix + "_{}".format(last_idx + 1)
                else:
                    identifier = identifier.lower()
                data = layer if idx == len(scopes) - 1 else None
                self.create_node(tag=scopes[idx], 
                                 identifier=identifier, 
                                 parent=parent,
                                 data=data)
                
    def update_hierarchical_order(self):
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
        
    def merge_node(self, sub_layers, scope_name):
        def get_inputs_outputs(layers):
            inputs = list()
            outputs = list()
            cur_outputs = list()
            layer_ids = list(layers.keys())
            for layer_id, layer in layers.items():
                # 获取输出节点名字
                if layer_id not in self.pd_graph.edges_out:
                    for output_name in layer.outputs:
                        if not output_name.startswith("x") or output_name in outputs \
                                or layer.kernel == "prim.assert":
                            continue
                        else:
                            outputs.append(output_name)
                else:
                    for out_layer_id in self.pd_graph.edges_out[layer_id]:
                        if out_layer_id not in layer_ids:
                            for output_name in layer.outputs:
                                if not output_name.startswith("x") or output_name in outputs:
                                    continue
                                else:
                                    outputs.append(output_name)
                # 获取输入节点名字
                for k, v in layer.inputs.items():
                    if v not in cur_outputs:
                        inputs.append(v)
                if ("paddle.nn" in layer.kernel and "functional" not in layer.kernel
                    ) or "paddle.fluid.dygraph" in layer.kernel:
                    cur_outputs.extend(layer.outputs[1:])
                else:
                    cur_outputs.extend(layer.outputs)
            return inputs, outputs
        
        module_count = dict()
        class_name_segs = scope_name.split("/")
        class_name_list = list()
        for name in class_name_segs:
            class_name_list.append(name[0].upper() + name[1:])
        class_name = "_".join(class_name_list)
        inputs, outputs = get_inputs_outputs(sub_layers)
        inputs_dict = dict()
        for i, input in enumerate(inputs):
            inputs_dict["input_{}".format(i)] = input
        sub_layers, name_new2old = rename_layers(sub_layers)
        use_params = False
        if class_name.lower() == self.pd_graph.name.lower():
            use_params = True
        code_str, param_prefix_list = gen_layer_code(self.pd_graph, sub_layers, class_name, use_params=use_params)
        self.codes.append(code_str)
        if class_name  not in module_count:
            module_count[class_name] = 0
        else:
            module_count[class_name] += 1
#         outputs = ["{}_{}".format(class_name.lower(), module_count[class_name])] + outputs
        outputs = ["{}".format(class_name.lower())] + outputs
        diff_attrs = dict() #TODO：获取相同子图后可能有形参不一致
        new_layer = PaddleLayer(id=list(sub_layers.keys())[-1],
                                kernel="module",
                                inputs=inputs_dict,
                                outputs=outputs,
                                scope_name=scope_name,
                                module=class_name,
                                **diff_attrs)
        for i, (layer_id, layer) in enumerate(sub_layers.items()):
            if i == len(sub_layers) - 1:
                self.pd_graph.layers[layer_id] = new_layer
            else:
                self.pd_graph.layers.pop(layer_id)
        self.pd_graph.build()
        return new_layer, name_new2old
                        
    def convert_subgraph_to_layer(self):
        depths = sorted(list(self._hierarchical_order.keys()), reverse=True)
        all_name_new2old = dict()
        for depth in depths[1:]:
            for node_name in self._hierarchical_order[depth]:
                node_inst = self[node_name]
                if node_inst.data is None:
                    sub_layers = dict()
                    for successor_name in node_inst.successors(self.identifier):
                        sub_layers[self[successor_name].data.id] = self[successor_name].data
                    new_layer, name_new2old = self.merge_node(sub_layers, node_name)
                    all_name_new2old.update(name_new2old)
                    node_inst.data = new_layer
        param_name_dict = self.rename_parameters()
        for seg, all_seg in param_name_dict.items():
            old = all_name_new2old[seg]
            new = all_seg
                    
    def rename_parameters(self):
        depths = sorted(list(self._hierarchical_order.keys()))
        param_name_dict = dict()
        module_count = dict()
        for node_name in self._hierarchical_order[depths[-1]]:
            node_inst = self[node_name]
            if node_inst.data.kernel in NN_KERNEL_WITH_PARAMS:
                param_prefix = node_inst.data.outputs[0]
                origin_param_prefix = param_prefix
                while node_inst.predecessor(self.identifier) != self.pd_graph.name:
                    param_prefix = node_inst.predecessor(self.identifier) + "." + param_prefix
                    node_inst = self[node_inst.predecessor(self.identifier)]
                param_name_dict[origin_param_prefix] = param_prefix
        return param_name_dict
                
    def save_source_files(self, save_dir):
        self.update_hierarchical_order()
        self.convert_subgraph_to_layer()
        import_list = ["import paddle",
                       "import paddle.fluid as fluid",
                       "",]
        import_str = "\n".join(import_list)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        f = open(osp.join(save_dir, 'x2paddle_code.py'), 'w')
        f.write(import_str)
        for code in self.codes:
            f.write(code)
            f.write("\n")
        f.close()
        self.rename_parameters()
