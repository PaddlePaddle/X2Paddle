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
from x2paddle.decoder.pytorch_pattern_proposal import *
from x2paddle.decoder import pytorch_pattern_proposal as generate_proposal


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
        
        
class PyTorchGraph(Graph):
    def __init__(self, pytorch_graph, params):
        self.params = params
        self.pytorch_graph = pytorch_graph
        super(PyTorchGraph, self).__init__(pytorch_graph)
        self.params = params
        self.nodeid2nodename = {}
        self.input_outputs = {}
        self.delete_nodes = []
        
    def _get_input_info(self, graph):
        for node in graph.nodes():
            _, node_ids = self._get_node_info(node)                    
            node_str = node.__str__()
            input_pattern = re.compile('[(]%.*?[)]')
            input_list_str = input_pattern.findall(node_str)
            if len(input_list_str) > 0:
                input_str = input_list_str[0][1: -1]
                input_node_ids = input_str.split(', ')
                for input_node_id in input_node_ids:
                    if input_node_id not in self.input_outputs:
                        self.input_outputs[input_node_id] = []
                    self.input_outputs[input_node_id].extend(node_ids)
                    self.input_outputs[input_node_id] = list(set(self.input_outputs[input_node_id]))
            if len(list(node.blocks())) != 0:
                for block in node.blocks():
                    self._get_input_info(block)
        
    def _get_node_info(self, node):
        node_str = node.__str__()
        node_str = node_str.split('\n')[0]
        match_pattern = re.compile(r'%.*? :')
        matches = match_pattern.findall(node_str)
        node_ids = []
        for m in matches:
            node_ids.append(m[0:-2])
        node_name = '_'.join(node_ids)
        self.nodeid2nodename[node_name] = node_ids
        return node_name, node_ids
            
            
    def deal_node(self, node):
        node_name, node_ids = self._get_node_info(node)
        if set(node_ids).issubset(self.delete_nodes):
            return 
        kind = node.kind()
        outs = find_pattern_part(node, self.pytorch_graph)
        if outs is not None:
            combined_node = getattr(generate_proposal, outs[0] + 'CombinedNode')(outs[1])
            node_ids = combined_node.get_node_ids()
            can_combined = True
            for node_id in node_ids:
                if node_id not in self.input_outputs or \
                        node_id in combined_node.node_name:
                    continue
                outputs = self.input_outputs[node_id]
                if not set(outputs).issubset(node_ids):
                    can_combined = False
                    break
            if can_combined:
                self.delete_nodes.extend(node_ids)  
                comined_node_name = '_'.join(combined_node.node_name)
                combined_node.input_names = []
                self.node_map[comined_node_name] = PyTorchGraphNode(combined_node, 
                                                            combined_node.kind, 
                                                            comined_node_name)
                node_attrs = []
                node_params = {}
                for input_node_name in combined_node.inputs:
                    match = [s for s in list(self.node_map.keys()) if input_node_name + '_%' in s or s.endswith(input_node_name)]
                    if input_node_name in self.node_map:
                        combined_node.input_names.append(input_node_name)
                        self.connect(input_node_name, comined_node_name)
                        node_attrs.append(None)
                    elif len(match) == 1:
                        combined_node.input_names.append(input_node_name)
                        self.connect(match[0], comined_node_name)
                        node_attrs.append(None)
                    else:
                        if input_node_name is None:
                            node_attrs.append(None)
                            continue
                        node_attrs.append(self.attrs[input_node_name])
                        if self.attrs[input_node_name] in list(self.params.keys()):
                            node_params[self.attrs[input_node_name]] = self.params[self.attrs[input_node_name]]
                self.node_map[comined_node_name].set_attrs(node_attrs)
                self.node_map[comined_node_name].set_params(node_params)
                return
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
            # TODO(syf)
            self.parse_if_node(node)
        elif kind.startswith('aten'):
            node_attrs = []
            node_params = {}
            kind = kind.split('::')[-1]
            self.node_map[node_name] = PyTorchGraphNode(node, kind, node_name)
            for index, input in enumerate(node.inputs()):
                input_node = input.node()
                if input_node.kind() == 'prim::Param':
                    input_node_name = '%' + input.debugName()
                else:
                    input_node_name, _ = self._get_node_info(input_node)
                if input_node_name in self.node_map:
                    self.connect(input_node_name, node_name)
                    node_attrs.append(None)
                else:
                    if input_node_name in self.attrs:
                        node_attrs.append(self.attrs[input_node_name])
                        if self.attrs[input_node_name] in list(self.params.keys()):
                            node_params[self.attrs[input_node_name]] = self.params[self.attrs[input_node_name]]
                    else:
                        node_attrs.append(input_node_name)
            self.node_map[node_name].set_attrs(node_attrs)
            self.node_map[node_name].set_params(node_params)
        else:
            print(node_name)
        
    def build(self):
        self.attrs = {}
        for node in self.pytorch_graph.inputs():
            node_id = '%' + node.debugName()
            if 'self' not in node_id:
                node = node.node()
                node = PyTorchGraphDataNode(node, 'data', node_id)
                self.node_map[node_id] = node
        self._get_input_info(self.pytorch_graph)
        for node in self.pytorch_graph.nodes():
            self.deal_node(node)
        super(PyTorchGraph, self).build()
        
    def get_input_node(self, pytorch_node, idx=0, copy=False):
        input_node_name = pytorch_node.inputs[idx]
        assert input_node_name in self.node_map, 'The {} isn\'t a valid node'.format(
            name)
        pytorch_node_input_names = []
        if pytorch_node.layer_type.startswith('torch'):
            pytorch_node_input_names = pytorch_node.layer.input_names
        else:
            node_str = pytorch_node.layer.__str__()
            pattern = re.compile(r"[(]%.*?[)]")
            m = pattern.findall(node_str)
            input_node_str = m[0]
            pytorch_node_input_names = []
            pytorch_node_input_names_all = input_node_str[1:-1].split(', ')
            for ipt in pytorch_node_input_names_all:
                match = [s for s in list(self.node_map.keys()) if ipt + '_%' in s or s.endswith(ipt)]
                if ipt in self.node_map:
                    pytorch_node_input_names.append(ipt)
                elif len(match) == 1:
                    pytorch_node_input_names.append(ipt)
        input_node = self.node_map[input_node_name]
        if input_node.layer_type.startswith('torch'):
            node_ids = input_node.layer.node_name
        else:
            _, node_ids = self._get_node_info(input_node.layer)
        if len(node_ids) > 1:
            need_idx = node_ids.index(pytorch_node_input_names[idx])
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
        