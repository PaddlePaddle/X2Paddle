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

import os
import re
import torch
import torch.jit
from torch.jit import _unique_state_dict
from x2paddle.core.graph import GraphNode, Graph
from x2paddle.core.fluid_code import FluidCode
from x2paddle.decoder.pytorch_combime_node import get_combined_graph, _get_str_line_index, CombinedNode, Invalid_BatchNormCombinedNode


class PyTorchGraphNode(GraphNode):
    def __init__(self, layer, layer_type, layer_name=None):
        assert layer_name is not None, 'The layer_name must be not None!'
        super(PyTorchGraphNode, self).__init__(
            layer,
            layer_name.replace('/', '_').replace('-', '_').replace(
                '.', '_').replace('%', 'x_'))
        self.layer_type = layer_type
        self.fluid_code = FluidCode()
        self.params = None
        self.attrs = None

    def set_params(self, params):
        self.params = params

    def set_attrs(self, attrs):
        self.attrs = attrs


class PyTorchGraphControlNode(GraphNode):
    def __init__(self, layer, layer_type, layer_name=None):
        assert layer_name is not None, 'The layer_name must be not None!'
        super(PyTorchGraphControlNode, self).__init__(
            layer,
            layer_name.replace('/', '_').replace('-', '_').replace(
                '.', '_').replace('%', 'x_'))
        self.layer_type = layer_type
        self.fluid_code = FluidCode()
        self.block_nodes_name = []
        self.attrs = None
        self.middle_name = None


class PyTorchGraph(Graph):
    def __init__(self,
                 pytorch_graph,
                 params,
                 origin_model,
                 graph_type='Normal',
                 graph_opts=None,
                 graph_ipts=None,
                 graph_name=None):
        self.params = params
        self.origin_model = origin_model
        self.graph_type = graph_type
        self.pytorch_graph = pytorch_graph
        self.graph_opts = graph_opts
        self.graph_ipts = graph_ipts
        self.graph_name = graph_name
        if graph_name is not None:
            self.layer_name = graph_name.split('__')[0].replace(
                '/', '_').replace('-', '_').replace('.', '_').replace('%', 'x_')
        super(PyTorchGraph, self).__init__(pytorch_graph)
        self.params = params
        self.attrs = {}
        self.ipt_opts = {}
        self.delete_indexes = []
        self.ifelse_id = 0
        self.loop_id = 0
        self.inputs = list()
        self.outputs = list()

    def _get_input_info(self, graph):
        for node in graph.nodes():
            _, node_ids = self._get_node_info(node)
            node_str = node.__str__()
            ipts_pattern = re.compile('[(]%.*?[)]')
            ipts_list_str = ipts_pattern.findall(node_str)
            if len(ipts_list_str) > 0:
                ipts_str = ipts_list_str[0][1:-1]
                ipt_id_list = ipts_str.split(', ')
                for ipt_id in ipt_id_list:
                    if ipt_id not in self.ipt_opts:
                        self.ipt_opts[ipt_id] = []
                    self.ipt_opts[ipt_id].extend(node_ids)
                    self.ipt_opts[ipt_id] = list(set(self.ipt_opts[ipt_id]))
            if len(list(node.blocks())) != 0:
                for block in node.blocks():
                    self._get_input_info(block)

    def _get_node_info(self, node):
        node_str = node.__str__()
        node_str = node_str.split('\n')[0]
        info_pattern = re.compile(r'%.*? :')
        matches = info_pattern.findall(node_str)
        node_ids = []
        for m in matches:
            node_ids.append(m[0:-2])
        node_name = '_'.join(node_ids)
        return node_name, node_ids

    def _get_ipt_node_name(self, ipt_id):
        match = [s for s in list(self.node_map.keys()) \
                if ipt_id + '_%' in s or s.endswith(ipt_id) or ipt_id + '__' in s]
        if ipt_id in self.node_map:
            return ipt_id
        elif len(match) >= 1:
            return match[-1]
        elif ipt_id in self.attrs:
            return False
        elif self.father_node_map is not None:
            father_match = [s for s in list(self.father_node_map.keys()) \
                            if ipt_id + '_%' in s or s.endswith(ipt_id) or ipt_id + '__' in s]
            if ipt_id in self.father_node_map:
                return ipt_id
            elif len(father_match) >= 1:
                return father_match[-1]
            elif ipt_id in self.father_attrs:
                return False

    def deal_node(self, node):
        node_name, node_ids = self._get_node_info(node)
        kind = node.kind()
        node_str = node.__str__().replace('\n', '')
        if kind == 'prim::If' or kind == 'prim::Loop':
            first_line = node.__str__().split('\n')[0]
            node_str = first_line.lstrip()
        index = self.line_index[node_str]
        if index in self.line_combine_info:
            self.delete_indexes.extend(
                list(range(index, index + self.line_combine_info[index][0])))
            cnode = self.line_combine_info[index][1]
            if isinstance(cnode, Invalid_BatchNormCombinedNode):
                return False
            cnode_name = '_'.join(cnode.cnode_ids)
            cnode.input_ids = []
            self.node_map[cnode_name] = PyTorchGraphNode(cnode, cnode.kind,
                                                         cnode_name)
            node_attrs = []
            node_params = {}
            for ipt_id in cnode.inputs:
                match = [
                    s for s in list(self.node_map.keys())
                    if ipt_id + '_%' in s or s.endswith(ipt_id) or ipt_id + '__'
                    in s
                ]
                if ipt_id in self.node_map:
                    cnode.input_ids.append(ipt_id)
                    self.connect(ipt_id, cnode_name)
                    if ipt_id in self.attrs:
                        node_attrs.append(self.attrs[ipt_id])
                    else:
                        node_attrs.append(None)
                elif len(match) >= 1:
                    cnode.input_ids.append(ipt_id)
                    self.connect(match[-1], cnode_name)
                    if ipt_id in self.attrs:
                        node_attrs.append(self.attrs[ipt_id])
                    else:
                        node_attrs.append(None)
                elif ipt_id is None or ipt_id in self.attrs:
                    cnode.input_ids.append(ipt_id)
                    if ipt_id is None:
                        node_attrs.append(None)
                        continue
                    node_attrs.append(self.attrs[ipt_id])
                    if self.attrs[ipt_id] in list(self.params.keys()):
                        node_params[self.attrs[ipt_id]] = self.params[
                            self.attrs[ipt_id]]
                elif self.father_node_map is not None:
                    father_match = [
                        s for s in list(self.father_node_map.keys())
                        if ipt_id + '_%' in s or s.endswith(ipt_id) or ipt_id +
                        '__' in s
                    ]
                    if ipt_id in self.father_node_map or len(father_match) >= 1:
                        cnode.input_ids.append(ipt_id)
                        if ipt_id in self.father_attrs:
                            node_attrs.append(self.father_attrs[ipt_id])
                        else:
                            node_attrs.append(None)
                        # 若有节点的输入节点在子图外，则该节点为子图的father_input_nodes
                        if ipt_id in self.father_node_map:
                            self.father_input_nodes.append(ipt_id)
                        else:
                            self.father_input_nodes.append(father_match[-1])
                    else:
                        cnode.input_ids.append(ipt_id)
                        if ipt_id is None:
                            node_attrs.append(None)
                            continue
                        node_attrs.append(self.father_attrs[ipt_id])
                        if self.father_attrs[ipt_id] in list(self.params.keys(
                        )):
                            node_params[self.father_attrs[
                                ipt_id]] = self.params[self.father_attrs[
                                    ipt_id]]
                else:
                    print('Error----', ipt_id)
            self.node_map[cnode_name].set_attrs(node_attrs)
            self.node_map[cnode_name].set_params(node_params)
        else:
            if index in self.delete_indexes:
                return False
            if kind == 'prim::GetAttr':
                match_pattern = re.compile(r'name=\"(.*)\"')
                matches = match_pattern.findall(node.__str__())
                attr_name_list = [matches[0]]
                while True:
                    if len(list(node.inputs())) == 0:
                        break
                    input_node = list(node.inputs())[0].node()
                    matches = match_pattern.findall(input_node.__str__())
                    if len(matches) == 0:
                        break
                    attr_name = matches[0]
                    attr_name_list.insert(0, attr_name)
                    node = input_node
                self.attrs[node_name] = '.'.join(attr_name_list)
            elif kind == 'prim::Constant':
                match_pattern1 = re.compile(r': (.*) = prim::Constant')
                matches1 = match_pattern1.findall(node.__str__())
                match_pattern2 = re.compile(r'value=(.*)]')
                matches2 = match_pattern2.findall(node.__str__())
                if len(matches2) == 0:
                    self.attrs[node_name] = None
                else:
                    if matches1[0] == 'int':
                        self.attrs[node_name] = int(matches2[0])
                    elif matches1[0] == 'float':
                        self.attrs[node_name] = float(matches2[0])
                    elif matches1[0] == 'bool':
                        self.attrs[node_name] = bool(matches2[0])
                    else:
                        raise Exception('The {} is not implement yet!'.format(
                            matches1[0]))
            elif kind == 'prim::ListConstruct':
                if len(list(node.inputs())) == 0:
                    self.attrs[node_name] = []
                else:
                    self.node_map[node_name] = PyTorchGraphNode(node, "list_construct",
                                                             node_name)
                    self.node_map[node_name].node_ids = node_ids
                    node_attrs = []
                    is_all_attr = True
                    for input_node in node.inputs():
                        input_node = input_node.node()
                        input_node_name, _ = self._get_node_info(input_node)
                        if input_node_name in self.node_map:
                            self.connect(input_node_name, node_name)
                            node_attrs.append(input_node_name)
                            is_all_attr = False
                        elif input_node_name in self.attrs:
                            node_attrs.append(self.attrs[input_node_name])
                        elif input_node_name in self.father_node_map:
                            self.father_input_nodes.append(input_node_name)
                            node_attrs.append(input_node_name)
                            is_all_attr = False
                        elif input_node_name in self.father_attrs:
                            node_attrs.append(self.father_attrs[input_node_name])
                        else:
                            print('Error----', ipt_id)
                    if is_all_attr:
                        self.node_map.pop(node_name)
                        self.attrs[node_name] = node_attrs
                    else:
                        self.node_map[node_name].set_attrs(node_attrs)
            elif kind == 'prim::shape':
                input_node = list(node.inputs())[0]
                self.node_map[node_name] = PyTorchGraphNode(node, "shape",
                                                             node_name)
                self.node_map[node_name].node_ids = node_ids
                input_node_name, _ = self._get_node_info(input_node)
                if input_node_name in self.node_map:
                    self.connect(input_node_name, node_name)
                elif input_node_name in self.father_node_map:
                    self.father_input_nodes.append(input_node_name)
                else:
                    print('Error----', ipt_id)
            elif kind == 'prim::If':
                ifelse_node_name, ifelse_ids = self._get_node_info(node)
                if ifelse_node_name == '':
                    ifelse_node_name = '%ifelse' + str(self.ifelse_id)
                    self.ifelse_id += 1
                # 处理if这个分支
                control_node_name = ifelse_node_name + '__if'
                self.node_map[control_node_name] = PyTorchGraphControlNode(
                    node, 'control_if', control_node_name)
                self.node_map[control_node_name].node_ids = [control_node_name]
                ipt_id = '%' + list(node.inputs())[0].__str__().split(' ')[0]
                if not self._get_ipt_node_name(ipt_id):
                    self.node_map[ipt_id] = PyTorchGraphNode(None, "assign",
                                                             ipt_id)
                    self.node_map[ipt_id].node_ids = [ipt_id]
                    self.node_map[ipt_id].input_ids = None
                    if ipt_id in self.attrs:
                        if isinstance(self.attrs[ipt_id], str):
                            part_str = self.attrs[ipt_id].split('.')
                            ipt_attr = self.origin_model
                            for s in part_str:
                                ipt_attr = getattr(ipt_attr, s)
                        else:
                            ipt_attr = self.attrs[ipt_id]
                    else:
                        if isinstance(self.father_attrs[ipt_id], str):
                            part_str = self.father_attrs[ipt_id].split('.')
                            ipt_attr = self.origin_model
                            for s in part_str:
                                ipt_attr = getattr(ipt_attr, s)
                        else:
                            ipt_attr = self.father_attrs[ipt_id]
                    self.node_map[ipt_id].set_attrs([ipt_attr])
                    ipt_node_name = ipt_id
                else:
                    ipt_node_name = self._get_ipt_node_name(ipt_id)
                self.connect(ipt_node_name, control_node_name)
                block = list(node.blocks())[0]
                sub_graph_name = ifelse_node_name + '__block0'
                sub_graph = PyTorchGraph(
                    block,
                    self.params,
                    self.origin_model,
                    graph_type='If',
                    graph_opts=ifelse_ids,
                    graph_name=sub_graph_name)
                sub_graph.build(
                    self.line_index, self.line_combine_info, self.node_map
                    if self.father_node_map is None else
                    self.node_map.update(self.father_node_map), self.attrs
                    if self.attrs is not None else
                    self.attrs.update(self.father_attrs))
                self.node_map[sub_graph_name] = sub_graph
                sub_graph_ipts = sub_graph.father_input_nodes
                for ipt_name in sub_graph_ipts:
                    self.connect(ipt_name, sub_graph_name)
                    self.connect(ipt_name, control_node_name)
                self.connect(control_node_name, sub_graph_name)
                self.node_map[sub_graph_name].node_ids = ifelse_ids
                # 处理else这个分支
                control_node_name = ifelse_node_name + '__else'
                block = list(node.blocks())[1]
                if len(list(block.nodes())) == 0 and len(list(block.outputs())) == 0:
                    return True
                self.node_map[control_node_name] = PyTorchGraphControlNode(
                    node, 'control_else', control_node_name)
                self.node_map[control_node_name].node_ids = [control_node_name]
                self.connect(sub_graph_name, control_node_name)
                if len(list(block.nodes())) == 0:
                    return True
                sub_graph_name = ifelse_node_name + '__block1'
                sub_graph = PyTorchGraph(
                    block,
                    self.params,
                    self.origin_model,
                    graph_type='Else',
                    graph_opts=ifelse_ids,
                    graph_name=sub_graph_name)
                sub_graph.build(
                    self.line_index, self.line_combine_info, self.node_map
                    if self.father_node_map is None else
                    self.node_map.update(self.father_node_map), self.attrs
                    if self.attrs is not None else
                    self.attrs.update(self.father_attrs))
                self.node_map[sub_graph_name] = sub_graph
                sub_graph_ipts = sub_graph.father_input_nodes
                for ipt_name in sub_graph_ipts:
                    self.connect(ipt_name, sub_graph_name)
                self.connect(control_node_name, sub_graph_name)
                self.node_map[sub_graph_name].node_ids = ifelse_ids
            elif kind == 'prim::Loop':
                loop_node_name, loop_ids = self._get_node_info(node)
                if loop_node_name == '':
                    loop_node_name = '%loop' + str(self.loop_id)
                    self.loop_id += 1
                # 处理loop的block
                control_node_name = loop_node_name + '__loop'
                self.node_map[control_node_name] = PyTorchGraphControlNode(
                    node, 'control_loop', control_node_name)
                self.node_map[control_node_name].node_ids = [control_node_name]
                ipts = []
                for i, ipt_info in enumerate(node.inputs()):
                    if i == 1:
                        continue
                    ipt_id = '%' + ipt_info.__str__().split(' ')[0]
                    if not self._get_ipt_node_name(ipt_id):
                        # 输入不是一个节点，而是一个attr，所以需要使用assign对其分配名字
                        self.node_map[ipt_id] = PyTorchGraphNode(None, "assign",
                                                                 ipt_id)
                        self.node_map[ipt_id].node_ids = [ipt_id]
                        self.node_map[ipt_id].input_ids = None
                        if ipt_id in self.attrs:
                            if isinstance(self.attrs[ipt_id], str):
                                part_str = self.attrs[ipt_id].split('.')
                                ipt_attr = self.origin_model
                                for s in part_str:
                                    ipt_attr = getattr(ipt_attr, s)
                            else:
                                ipt_attr = self.attrs[ipt_id]
                        else:
                            if isinstance(self.father_attrs[ipt_id], str):
                                part_str = self.father_attrs[ipt_id].split('.')
                                ipt_attr = self.origin_model
                                for s in part_str:
                                    ipt_attr = getattr(ipt_attr, s)
                            else:
                                ipt_attr = self.father_attrs[ipt_id]
                        ipt_node_name = ipt_id
                        self.node_map[ipt_id].set_attrs([ipt_attr])
                    else:
                        ipt_node_name = self._get_ipt_node_name(ipt_id)
                    self.connect(ipt_node_name, control_node_name)
                    ipts.append(ipt_node_name)
                
                block = list(node.blocks())[0]
                sub_graph_name = loop_node_name + '__block'
                sub_graph = PyTorchGraph(
                    block,
                    self.params,
                    self.origin_model,
                    graph_type='Loop',
                    graph_opts=loop_ids,
                    graph_ipts=ipts,
                    graph_name=sub_graph_name)
                sub_graph.build(
                    self.line_index, self.line_combine_info, self.node_map
                    if self.father_node_map is None else
                    self.node_map.update(self.father_node_map), self.attrs
                    if self.attrs is not None else
                    self.attrs.update(self.father_attrs))
                self.node_map[sub_graph_name] = sub_graph
                sub_graph_ipts = sub_graph.father_input_nodes
                for ipt_name in sub_graph_ipts:
                    self.connect(ipt_name, sub_graph_name)
                self.connect(control_node_name, sub_graph_name)
                self.node_map[sub_graph_name].node_ids = loop_ids
            else:
                print('----', index)
                print(node_name)
        return True

    def build(self,
              line_index=None,
              line_combine_info=None,
              father_node_map=None,
              father_attrs=None):
        # 处理输入节点
        if line_index is None and line_combine_info is None:
            for node in self.pytorch_graph.inputs():
                node_id = '%' + node.debugName()
                if 'self' not in node_id:
                    node = node.node()
                    node = PyTorchGraphNode(node, 'data', node_id)
                    self.node_map[node_id] = node
                    self.node_map[node_id].node_ids = [node_id]
        elif self.graph_type == 'Loop':
            for i, node_info in enumerate(self.pytorch_graph.inputs()):
                node_id = '%' + node_info.__str__().split(' ')[0]
                if i == 0:
                    self.graph_name = self.graph_name.replace('__block',
                                                              '__loop')
                    ipt_name = self.graph_name
                else:
                    ipt_name = self.graph_ipts[i]
                self.node_map[node_id] = PyTorchGraphNode(None, "assign",
                                                          node_id)
                self.node_map[node_id].node_ids = [node_id]
                self.node_map[node_id].input_ids = [ipt_name]
                self.node_map[node_id].set_attrs([ipt_name])
                self.father_input_nodes.append(ipt_name)
        self.father_node_map = father_node_map
        self.father_attrs = father_attrs
        # 获取输入节点id与对应输出所有节点id所组成list的关系
        self._get_input_info(self.pytorch_graph)
        # 获取每一行字符串与行号的关系
        if line_index is None:
            line_index = _get_str_line_index(self.pytorch_graph.__str__())
            self.line_index = {}
            for k, v in line_index.items():
                self.line_index[k.lstrip()] = line_index[k]
        else:
            self.line_index = line_index
        # 获取多行组合信息
        if line_combine_info is None:
            self.line_combine_info = {}
            line_combine_info = get_combined_graph(self.pytorch_graph,
                                                   self.ipt_opts)
#             import pickle
#             with open('tmp.pickle', 'rb') as file:
#                 line_combine_info =pickle.load(file)
            for l in line_combine_info:
                self.line_combine_info.update(l)
        else:
            self.line_combine_info = line_combine_info
        # 处理每个节点
        for node in self.pytorch_graph.nodes():
            self.deal_node(node)
        # 处理return节点
        if hasattr(self.pytorch_graph, 'returnNode'):
            return_node = self.pytorch_graph.returnNode()
            return_node_ipts = list(return_node.inputs())
            if self.graph_opts is not None:
                if self.graph_type == 'If':
                    return_node_name = "_".join(self.graph_opts) + '__0'
                elif self.graph_type == 'Else':
                    return_node_name = "_".join(self.graph_opts) + '__1'
                else:
                    return_node_name = "_".join(self.graph_opts)
                self.node_map[return_node_name] = PyTorchGraphNode(
                    return_node, "assign", return_node_name)
                self.node_map[return_node_name].input_ids = None
                self.node_map[return_node_name].node_ids = []
                attrs = []
                for i, opt_id in enumerate(self.graph_opts):
                    start_id = 1 if self.graph_type == "Loop" else 0
                    ipt_id = '%' + return_node_ipts[start_id + i].__str__(
                    ).split(' ')[0]
                    match = [
                        s for s in list(self.node_map.keys())
                        if ipt_id + '_%' in s or s.endswith(ipt_id) or ipt_id +
                        '__' in s
                    ]
                    if ipt_id in self.node_map:
                        self.node_map[return_node_name].node_ids.append(opt_id)
                        self.connect(ipt_id, return_node_name)
                        attrs.append(ipt_id)
                    elif len(match) >= 1:
                        self.node_map[return_node_name].node_ids.append(opt_id)
                        self.connect(match[-1], return_node_name)
                        attrs.append(match[-1])
                    elif ipt_id is None or ipt_id in self.attrs:
                        self.node_map[return_node_name].node_ids.append(opt_id)
                        attrs.append(self.attrs[ipt_id])
                    elif self.father_node_map is not None:
                        father_match = [
                            s for s in list(self.father_node_map.keys())
                            if ipt_id + '_%' in s or s.endswith(ipt_id) or
                            ipt_id + '__' in s
                        ]
                        if ipt_id in self.father_node_map or len(
                                father_match) >= 1:
                            self.node_map[return_node_name].node_ids.append(
                                opt_id)
                            # 若有节点的输入节点在子图外，则该节点为子图的father_input_nodes
                            if ipt_id in self.father_node_map:
                                self.father_input_nodes.append(ipt_id)
                                attrs.append(ipt_id)
                            else:
                                self.father_input_nodes.append(father_match[-1])
                                attrs.append(father_match[-1])
                        else:
                            self.node_map[return_node_name].node_ids.append(
                                opt_id)
                            attrs.append(self.father_attrs[ipt_id])
                    else:
                        print('Error----', ipt_id)
                self.node_map[return_node_name].set_attrs(attrs)
        self.father_input_nodes = list(set(self.father_input_nodes))
        super(PyTorchGraph, self).build()

    def get_input_node(self, pytorch_node, idx=0, copy=False):
        # 获取输入node
        if len(pytorch_node.inputs) == 0 or \
           (hasattr(pytorch_node.layer, 'input_ids') and len(pytorch_node.layer.input_ids) != len(pytorch_node.inputs)):
            
            if isinstance(pytorch_node.layer, CombinedNode):
                ipt_id = pytorch_node.layer.input_ids[idx]
            else:
                if pytorch_node.layer is None and pytorch_node.input_ids is not None:
                    ipt_id = pytorch_node.input_ids[0]
                elif pytorch_node.layer is not None:
                    ipt_id = '%' + list(pytorch_node.layer.inputs())[
                        idx].__str__().split(' ')[0]
                else:
                    print('Error------', pytorch_node.layer_name)
            match = [s for s in list(self.node_map.keys()) \
                     if ipt_id + '_%' in s or s.endswith(ipt_id) or ipt_id + '__' in s]
            if len(match) > 1:
                ipt_name = match[-1]
                ipt_pytorch_node = self.node_map[ipt_name]
            elif ipt_id in self.node_map:
                ipt_name = ipt_id
                ipt_pytorch_node = self.node_map[ipt_name]
            elif self.father_node_map is not None:
                father_match = [s for s in list(self.father_node_map.keys()) \
                            if ipt_id + '_%' in s or s.endswith(ipt_id) or ipt_id + '__' in s]
                if len(father_match) > 1:
                    ipt_name = father_match[-1]
                    ipt_pytorch_node = self.father_node_map[ipt_name]
                elif ipt_id in self.father_node_map:
                    ipt_name = ipt_id
                    ipt_pytorch_node = self.father_node_map[ipt_name]
            else:
                print('Error------', ipt_id)
        else:
            ipt_name = pytorch_node.inputs[idx]
            if ipt_name in self.node_map:
                ipt_pytorch_node = self.node_map[ipt_name]
            else:
                ipt_pytorch_node = self.father_node_map[ipt_name]
        # 获取当前node的输入node_id
        if isinstance(pytorch_node.layer, CombinedNode):
            ipt_ids = pytorch_node.layer.input_ids
        else:
            ipt_ids = []
            for ipt_layer in pytorch_node.layer.inputs():
                ipt_ids.append('%' + ipt_layer.__str__().split(' ')[0])
        # 解析输入node_id是输入node的第几个输出
        if hasattr(ipt_pytorch_node, 'layer') and isinstance(
                ipt_pytorch_node.layer, CombinedNode):
            ipt_pytorch_node_ids = ipt_pytorch_node.layer.cnode_ids
        else:
            ipt_pytorch_node_ids = ipt_pytorch_node.node_ids
        if len(ipt_pytorch_node_ids) > 1:
            need_idx = ipt_pytorch_node_ids.index(ipt_ids[idx])
            name = ipt_name + ':' + str(need_idx)
        else:
            name = ipt_name
        ipt_pytorch_node = self.get_node(name, copy=copy)
        
        
        if hasattr(
                ipt_pytorch_node,
                'layer') and ipt_pytorch_node.layer_type == 'data' and hasattr(
                    ipt_pytorch_node, 'index'):
            delattr(ipt_pytorch_node, 'index')
        return ipt_pytorch_node


class PyTorchDecoder(object):
    def __init__(self, model_path):
        try:
            model = torch.load(model_path)
        except:
            model = torch.load(model_path, map_location='cpu')
        self.params = _unique_state_dict(model)
        jit_script = self.get_jit_script(model)
        self.pytorch_graph = PyTorchGraph(jit_script, self.params, model)
        self.pytorch_graph.build()

    def get_jit_script(self, model):
        script = torch.jit.script(model)
        graph = self._optimize_graph(script.inlined_graph, False, True)
        return graph

    def _optimize_graph(self, graph, aten, export_raw_ir=False):
        # run dce first to eliminate dead parts of the graph that might have been
        # left behind by things like symbolic_override
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)

        torch._C._jit_pass_peephole(graph)
        torch._C._jit_pass_lint(graph)
        if not export_raw_ir:
            graph = torch._C._jit_pass_onnx(graph, aten)
            torch._C._jit_pass_lint(graph)
            torch._C._jit_pass_onnx_peephole(graph)
            torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        graph = torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        return graph
