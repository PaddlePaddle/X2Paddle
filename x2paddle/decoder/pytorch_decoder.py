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
from x2paddle.decoder.pytorch_combime_node import get_combined_graph, _get_str_line_index, CombinedNode


class PyTorchGraphNode(GraphNode):
    def __init__(self, layer, layer_type, layer_name=None):
        assert layer_name is not None, 'The layer_name must be not None!'
        super(PyTorchGraphNode, self).__init__(layer, 
                                               layer_name.replace('/', '_').replace('-', '_').replace('.', '_').replace('%', 'x_'))
        self.layer_type = layer_type
        self.fluid_code = FluidCode()
        self.params = None
        self.attrs = None

    def set_params(self, params):
        self.params = params
        
    def set_attrs(self, attrs):
        self.attrs = attrs


class PyTorchGraphDataNode(GraphNode):
    def __init__(self, layer, layer_type, layer_name=None):
        assert layer_name is not None, 'The layer_name must be not None!'
        super(PyTorchGraphDataNode, self).__init__(layer, 
                                                   layer_name.replace('/', '_').replace('-', '_').replace('.', '_').replace('%', 'x_'))
        self.layer_type = layer_type
        self.fluid_code = FluidCode()
        
        
class PyTorchGraphControlNode(GraphNode):
    def __init__(self, layer, layer_type, layer_name=None):
        assert layer_name is not None, 'The layer_name must be not None!'
        super(PyTorchGraphControlNode, self).__init__(layer, 
                                                   layer_name.replace('/', '_').replace('-', '_').replace('.', '_').replace('%', 'x_'))
        self.layer_type = layer_type
        self.fluid_code = FluidCode()
        self.block_nodes_name = []
        self.input_ids = []
        self.node_ids = []
        self.attrs = None
        
        
class PyTorchGraph(Graph):
    def __init__(self, pytorch_graph, params):
        self.params = params
        self.pytorch_graph = pytorch_graph
        super(PyTorchGraph, self).__init__(pytorch_graph)
        self.params = params
        self.ipt_opts = {}
        self.delete_indexes = []
        self.ifelse_id = 0
        
        
    def _get_input_info(self, graph):
        for node in graph.nodes():
            _, node_ids = self._get_node_info(node)                    
            node_str = node.__str__()
            ipts_pattern = re.compile('[(]%.*?[)]')
            ipts_list_str = ipts_pattern.findall(node_str)
            if len(ipts_list_str) > 0:
                ipts_str = ipts_list_str[0][1: -1]
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
  
            
    def deal_node(self, node):
        node_name, node_ids = self._get_node_info(node)
        kind = node.kind()
        node_str = node.__str__().replace('\n', '')
        if kind == 'prim::If' or kind == 'prim::Loop':
            first_line = node.__str__().split('\n')[0]
            node_str = first_line.lstrip()
        index = self.line_index[node_str]
        if index in self.line_combine_info:
            self.delete_indexes.extend(list(range(index, index + self.line_combine_info[index][0])))
            cnode = self.line_combine_info[index][1]
            cnode_name = '_'.join(cnode.node_name)
            cnode.input_names = []
            self.node_map[cnode_name] = PyTorchGraphNode(cnode, 
                                                        cnode.kind, 
                                                        cnode_name)
            node_attrs = []
            node_params = {}
            for input_node_name in cnode.inputs:
                match = [s for s in list(self.node_map.keys()) if input_node_name + '_%' in s or s.endswith(input_node_name)]
                if input_node_name in self.node_map:
                    cnode.input_names.append(input_node_name)
                    self.connect(input_node_name, cnode_name)
                    node_attrs.append(None)
                    self._connect_ifelse_block(input_node_name, cnode_name)
                elif len(match) == 1:
                    cnode.input_names.append(input_node_name)
                    self.connect(match[0], cnode_name)
                    node_attrs.append(None)
                    self._connect_ifelse_block(match[0], cnode_name)
                else:
                    if input_node_name is None:
                        node_attrs.append(None)
                        continue
                    node_attrs.append(self.attrs[input_node_name])
                    if self.attrs[input_node_name] in list(self.params.keys()):
                        node_params[self.attrs[input_node_name]] = self.params[self.attrs[input_node_name]]
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
                        raise Exception('The {} is not implement yet!'.format(matches1[0]))
            elif kind == 'prim::ListConstruct':
                list_info = []
                if len(list(node.inputs())) > 0:
                    for input_node in node.inputs():
                        input_node = input_node.node()
                        input_node_name, _ = self._get_node_info(input_node)
                        list_info.append(self.attrs[input_node_name])
                self.attrs[node_name] = list_info
            elif kind == 'prim::If':
                block_node_name, block_ids = self._get_node_info(node)
                if block_node_name == '':
                    block_node_name = '%ifelse' + str(self.ifelse_id)
                    self.ifelse_id += 1
                control_node_name = block_node_name
                self.node_map[control_node_name] = PyTorchGraphControlNode(node,
                                                                           'control_if',
                                                                           control_node_name)
                input = list(node.inputs())[0]
                input_node = input.node()
                if input_node.kind() == 'prim::Param':
                    input_node_name = '%' + input.debugName()
                else:
                    input_node_name, _ = self._get_node_info(input_node)
                self.connect(input_node_name, control_node_name)
                p = re.compile(r"prim::If[(]%(.*)[)]")
                m = p.search(node_str) 
                self.node_map[control_node_name].input_ids.append('%' + m.groups()[0]) 
                self.node_map[control_node_name].node_ids.append(control_node_name)
                for i, block in enumerate(node.blocks()):
                    block_opts_name = {}
                    for opt in block.outputs():
                        opt_name, opt_ids = self._get_node_info(opt.node())
                        for id in opt_ids:
                            block_opts_name[id] = opt_name
                    if i == 1: # else的block
                        last_node_name = list(self.node_map.keys())[-1]
                        control_node_name = block_node_name + '__else'
                        self.node_map[control_node_name] = PyTorchGraphControlNode(node,
                                                                           'control_else',
                                                                           control_node_name)
                        self.connect(last_node_name, control_node_name)
                    if len(list(block.nodes())) == 0:
                        if i == 0:
                            return_str = block.returnNode().__str__()
                            p = re.compile(r"prim::Return[(]%(.*)[)]")
                            m = p.search(return_str)  
                            assert m is not None, 'This is not a standard model'
                            self.node_map[block_node_name + '__0'] = PyTorchGraphControlNode(None, 
                                                                                       "assign", 
                                                                                       block_node_name + '__0')
                            
                            
                            p = re.compile(r"prim::Return[(](.*)[)]")
                            m = p.search(return_str)        
                            for nid in m.groups()[0].split(','):
                                self.node_map[block_node_name + '__0'].input_ids.append(nid)
                                self.connect(block_opts_name[nid], block_node_name + '__0')
                            self.connect(control_node_name, block_node_name + '__0')
                            self.node_map[control_node_name].block_nodes_name.append(block_node_name + '__0')
                            self.node_map[block_node_name + '__0'].attrs = block_ids
                            self.node_map[block_node_name + '__0'].node_ids.append(block_node_name + '__0')                       
                            last_node_name = block_node_name + '__0'
                        if i == 1:
                            return_str = block.returnNode().__str__()
                            p = re.compile(r"prim::Return[(]%(.*)[)]")
                            m = p.search(return_str)  
                            assert m is not None, 'This is not a standard model'
                            self.node_map[block_node_name + '__1'] = PyTorchGraphControlNode(None, 
                                                                                       "assign", 
                                                                                       block_node_name + '__1')
                            p = re.compile(r"prim::Return[(](.*)[)]")
                            m = p.search(return_str)        
                            for nid in m.groups()[0].split(','):
                                self.node_map[block_node_name + '__1'].input_ids.append(nid)
                                self.connect(block_opts_name[nid], block_node_name + '__1')
                            self.connect(control_node_name, block_node_name + '__1')
                            self.node_map[control_node_name].block_nodes_name.append(block_node_name + '__1')
                            self.node_map[block_node_name + '__1'].attrs = block_ids
                            self.node_map[block_node_name + '__1'].node_ids.append(block_node_name + '__1')
                            last_node_name = block_node_name + '__1'
                                
                    for j, node in enumerate(block.nodes()):
                        out = self.deal_node(node)
                        node_name = list(self.node_map.keys())[-1]
                        if node_name == control_node_name:
                            continue
                        if out:
                            self.node_map[control_node_name].block_nodes_name.append(node_name)
                            self.connect(control_node_name, node_name)
                        if i == 0 and j == len(list(block.nodes())) - 1:
                            return_str = block.returnNode().__str__()
                            p = re.compile(r"prim::Return[(]%(.*)[)]")
                            m = p.search(return_str)  
                            if m is None:
                                last_node_name = node_name
                            else:
                                self.node_map[block_node_name + '__0'] = PyTorchGraphControlNode(None, 
                                                                                           "assign", 
                                                                                           block_node_name + '__0')               
                                p = re.compile(r"prim::Return[(](.*)[)]")
                                m = p.search(return_str)        
                                for nid in m.groups()[0].split(','):
                                    self.node_map[block_node_name + '__0'].input_ids.append(nid)
                                    self.connect(block_opts_name[nid], block_node_name + '__0')
                                self.connect(control_node_name, block_node_name + '__0')
                                self.node_map[control_node_name].block_nodes_name.append(block_node_name + '__0')
                                self.node_map[block_node_name + '__0'].attrs = block_ids
                                self.node_map[block_node_name + '__0'].node_ids.append(block_node_name + '__0')
                                last_node_name = block_node_name + '__0'
                        elif i == 1 and j == len(list(block.nodes())) - 1:
                            return_str = block.returnNode().__str__()
                            p = re.compile(r"prim::Return[(]%(.*)[)]")
                            m = p.search(return_str)  
                            if m is not None:
                                self.node_map[block_node_name + '__1'] = PyTorchGraphControlNode(None, 
                                                                                           "assign", 
                                                                                           block_node_name + '__1')
                                p = re.compile(r"prim::Return[(](.*)[)]")
                                m = p.search(return_str)        
                                for nid in m.groups()[0].split(','):
                                    self.node_map[block_node_name + '__1'].input_ids.append(nid)
                                    self.connect(block_opts_name[nid], block_node_name + '__1')
                                self.connect(control_node_name, block_node_name + '__1')
                                self.node_map[control_node_name].block_nodes_name.append(block_node_name + '__1')
                                self.node_map[block_node_name + '__1'].attrs = block_ids
                                self.node_map[block_node_name + '__1'].node_ids.append(block_node_name + '__1')
            else:
                print(index)
                print(node_name)        
        return True
       
        
    def build(self):
        self.attrs = {}
        for node in self.pytorch_graph.inputs():
            node_id = '%' + node.debugName()
            if 'self' not in node_id:
                node = node.node()
                node = PyTorchGraphDataNode(node, 'data', node_id)
                self.node_map[node_id] = node
        self._get_input_info(self.pytorch_graph)
        line_index = _get_str_line_index(self.pytorch_graph.__str__())
        self.line_index = {}
        for k, v in line_index.items():
            self.line_index[k.lstrip()] = line_index[k]
        line_combine_info = get_combined_graph(self.pytorch_graph, self.ipt_opts)
        self.line_combine_info = {}
        for l in line_combine_info:
            self.line_combine_info.update(l)
        for node in self.pytorch_graph.nodes():
            self.deal_node(node)
        super(PyTorchGraph, self).build()
        
    def get_input_node(self, pytorch_node, idx=0, copy=False):
        input_node_name = pytorch_node.inputs[idx]
        assert input_node_name in self.node_map, 'The {} isn\'t a valid node'.format(
            name)
        input_node = self.node_map[input_node_name]
        if isinstance(pytorch_node.layer, CombinedNode):
            pytorch_node_input_ids = pytorch_node.layer.inputs
        else:
            pytorch_node_input_ids = pytorch_node.input_ids
        if isinstance(input_node.layer, CombinedNode):
            node_ids = input_node.layer.node_name
        else:
            if isinstance(input_node, PyTorchGraphControlNode):
                node_ids = input_node.node_ids
            else:
                _, node_ids = self._get_node_info(input_node.layer)                
        if len(node_ids) > 1:
            need_idx = node_ids.index(pytorch_node_input_ids[idx])
            name = input_node_name + ':' + str(need_idx)
        else:
            name = input_node_name
        input_pytorch_node = self.get_node(name, copy=copy)
        if isinstance(input_pytorch_node, PyTorchGraphDataNode) and hasattr(input_pytorch_node, 'index'):
            delattr(input_pytorch_node, 'index')
        return input_pytorch_node
  
        
class PyTorchDecoder(object):
    def __init__(self, model_path):
        try:
            model = torch.load(model_path)
        except:
            model = torch.load(model_path, map_location='cpu')
        self.params = _unique_state_dict(model)
        jit_script = self.get_jit_script(model)
        self.pytorch_graph = PyTorchGraph(jit_script, self.params)
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
        