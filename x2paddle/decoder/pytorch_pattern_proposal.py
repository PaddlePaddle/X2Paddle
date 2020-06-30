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

import re
import torch

patterns = {}
patterns['AdaptiveAvgPool2d'] = r"(\s*)%.*: int\[\] = aten::size[(]%.*[)] # .*(\n)(\s*)%.*: bool = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: int = prim::Constant\[value=.*\][(][)](\n)(\s*)%.*: str = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: int = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: int = aten::len[(]%.*[)] # .*(\n)(\s*)%.*: int = aten::len[(]%.*[)] # .*(\n)(\s*)%.*: bool = aten::le[(]%.*, %.*[)] # .*(\n)(\s*)= prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)= prim::RaiseException[(]%.*[)] # .*(\n)(\s*)-> [(][)](\n)(\s*)block1[(][)]:(\n)(\s*)-> [(][)](\n)(\s*)%.*: int\[\] = prim::ListConstruct[(][)](\n)(\s*).*: int = aten::len[(]%.*[)] # .*(\n)(\s*)%.*: int = aten::neg[(]%.*[)] # .*(\n)(\s*)%.*: int\[\] = aten::slice[(]%.*, %.*, %.*, %.*[)] # .*(\n)(\s*)%.*: int = aten::len[(]%.*[)] # .*(\n)(\s*)%.*: int = aten::len[(]%.*[)] # .*(\n)(\s*)%.*: int\[\] = prim::ListConstruct[(]%.*, %.*[)](\n)(\s*)%.*: int = prim::min[(]%.*[)] # .*(\n)(\s*)= prim::Loop[(]%.*, %.*[)] # .*(\n)(\s*)block0[(]%.* : int[)]:(\n)(\s*)%.*: int = aten::__getitem__[(]%.*, %.*[)] # .*(\n)(\s*)%.*: int\[\] = aten::append[(]%.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)(\s*)%.*: Tensor = aten::adaptive_avg_pool2d[(]%.*, %.*[)] # .*"

patterns['Dropout'] = r"(\s*)%.*: bool = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: float = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: bool = prim::GetAttr\[name=\".*\"\][(]%.*[)](\n)(\s*)%.*: str = prim::Constant\[value=\".*\"\][(][)] # .*(\n)(\s*)%.*: bool = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: float = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: float = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: bool = aten::lt[(]%.*, %.*[)] # .*(\n)(\s*)%.*: bool = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)-> [(]%.*[)](\n)(\s*)block1[(][)]:(\n)(\s*)%.*: bool = aten::gt[(]%.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)(\s*)= prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)= prim::RaiseException[(]%.*[)] # .*(\n)(\s*)-> [(][)](\n)(\s*)block1[(][)]:(\n)(\s*)-> [(][)](\n)(\s*)%.*: Tensor = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: Tensor = aten::dropout_[(]%.*, %.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)(\s*)block1[(][)]:(\n)(\s*)%.*: Tensor = aten::dropout[(]%.*, %.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)]"

patterns['Linear'] = r"(\s*)%.*: int = prim::Constant\[value=.*\][(][)](\n)(\s*)%.*: int = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: int = aten::dim[(]%.*[)] # .*(\n)(\s*)%.*: bool = aten::eq[(]%.*, %.*[)] # .*(\n)(\s*)%.*: Tensor = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: Tensor = aten::t[(]%.*[)] # .*(\n)(\s*)%.*: Tensor = aten::addmm[(]%.*, %.*, %.*, %.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)(\s*)block1[(][)]:(\n)(\s*)%.*: Tensor = aten::t[(]%.*[)] # .*(\n)(\s*)%.*: Tensor = aten::matmul[(]%.*, %.*[)] # .*(\n)(\s*)%.*: bool = prim::Constant\[value=.*\][(][)](\n)(\s*)%.*: Tensor = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: Tensor = aten::add_[(]%.*, %.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)(\s*)block1[(][)]:(\n)(\s*)-> [(]%.*[)](\n)(\s*)-> [(]%.*[)]"

patterns['MaxPool2d'] = r"(\s*)%.*: bool = prim::Constant\[value=.*\][(][)](\n)(\s*)%.*: int\[\] = prim::If[(].*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: int\[\] = prim::ListConstruct[(][)](\n)(\s*)-> [(](.*)[)](\n)(\s*)block1[(][)]:(\n)(\s*)-> [(]%.*[)](\n)(\s*)%.*: Tensor = aten::max_pool2d[(]%.*, %.*, %.*, %.*, %.*, %.*[)] # .*"

patterns['ReLU'] = r"(\s*)%.*: Tensor = prim::If(.*) # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: Tensor = aten::relu_[(].*[)] # .*(\n)(\s*)-> [(](.*)[)](\n)(\s*)block1[(][)]:(\n)(\s*)%.* : Tensor = aten::relu[(].*[)] # .*(\n)(\s*)-> [(](.*)[)]"


                                             
# TODO:
# 该部分代码后期要指定不同的匹配方式
# 通过修改patterns中key的顺序即可组合出不同的方案
def find_pattern_part(node, graph):
    node_str = node.__str__().replace('\n', '\n  ')
    graph_str = graph.__str__()
    need_graph_str = graph_str.split(node_str)[1]
    need_graph_str = node.__str__() + need_graph_str
    for key, pattern in patterns.items():
        m = re.match(pattern, need_graph_str)
        if m is None:
            continue
        else:
            return key, m.group()
    return None

class CombinedNode:
    def __init__(self, node_name, kind, inputs):        
        self.node_name = node_name
        self.kind = kind
        self.inputs = inputs
        
    def get_node_ids(self):
        node_ids = []
        lines = self.nodes_str.split('/n')
        for line in lines:
            pattern = re.compile(r"%.*?:")
            ms = pattern.findall(self.nodes_str)
            for m in ms:
                if ')' in m:
                    continue
                node_ids.append(m[:-2])
        return node_ids


class ReLUCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(ReLUCombinedNode, self).__init__(self.node_name,
                                               'torch_relu',
                                               self.inputs)
                                                       
    def get_combined_node_info(self):
        pattern1 = re.compile(r"%(.*) : Tensor = prim::If[(](.*)[)] #")
        m1 = pattern1.search(self.nodes_str)
        self.node_name = ['%' + m1.groups()[0]]
        pattern2 = re.compile(r"Tensor = aten::relu_[(]%(.*?)[)]")
        m2 = pattern2.search(self.nodes_str)
        self.inputs.append(m2.groups()[0])
        self.inputs.append(m1.groups()[1])
        
        
class AdaptiveAvgPool2dCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(AdaptiveAvgPool2dCombinedNode, self).__init__(self.node_name,
                                               'torch_adaptive_avg_pool2d',
                                               self.inputs)
        
    def get_combined_node_info(self):
        pattern1 = re.compile(r"%(.*) : Tensor = aten::adaptive_avg_pool2d[(]%(.*), %(.*)[)] #")
        m1 = pattern1.search(self.nodes_str)        
        self.node_name = ['%' + m1.groups()[0]]
        self.inputs.append('%' + m1.groups()[1])
        pattern2 = re.compile(r"%(.*) : int = aten::__getitem__[(]%(.*), %(.*)[)] #")
        m2 = pattern2.search(self.nodes_str)
        self.inputs.append('%' + m2.groups()[1])
        
        
class DropoutCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(DropoutCombinedNode, self).__init__(self.node_name,
                                                  'torch_dropout',
                                                  self.inputs)
        
    def get_combined_node_info(self):
        pattern = re.compile(r"%(.*) : Tensor = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.* : Tensor = aten::dropout_[(]%(.*), %(.*), %(.*)[)] #")
        m = pattern.search(self.nodes_str)
        self.node_name = ['%' + m.groups()[0]]
        self.inputs.append('%' + m.groups()[5])
        

class MaxPool2dCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(MaxPool2dCombinedNode, self).__init__(self.node_name,
                                                  'torch_max_pool2d',
                                                  self.inputs)
  
    def get_combined_node_info(self):
        pattern1 = re.compile(r"bool = prim::Constant\[value=(.*)\]")
        m1 = pattern1.search(self.nodes_str)
        stride = None
        if int(m1.groups()[0]) == 0:
            pattern2 = re.compile(r"block1[(][)]:(\n)(\s*)-> [(]%(.*)[)]")
            m2 = pattern2.search(self.nodes_str)
            stride = '%' + m2.groups()[2]
        pattern3 = re.compile(r"%(.*) : Tensor = aten::max_pool2d[(]%(.*), %(.*), %(.*), %(.*), %(.*), %(.*)[)] #")
        m3 = pattern3.search(self.nodes_str)
        self.node_name = ['%' + m3.groups()[0]]
        self.inputs.append('%' + m3.groups()[1])
        self.inputs.append('%' + m3.groups()[2])
        self.inputs.append(stride)
        self.inputs.append('%' + m3.groups()[4])
        self.inputs.append('%' + m3.groups()[5])
        self.inputs.append('%' + m3.groups()[6])
        
        
class LinearCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(LinearCombinedNode, self).__init__(self.node_name,
                                                  'torch_linear',
                                                  self.inputs)
        
    def get_combined_node_info(self):
        pattern1 = re.compile(r"%(.*) : Tensor = aten::matmul[(]%(.*?), %(.*?)[)] #")
        m1 = pattern1.search(self.nodes_str)
        self.inputs.append('%' + m1.groups()[1])
        pattern2 = re.compile(r"%(.*) : Tensor = aten::t[(]%(.*)[)] #")
        m2 = pattern2.search(self.nodes_str)
        self.inputs.append('%' + m2.groups()[1])
        pattern3 = re.compile(r"%(.*) : Tensor = aten::add_[(]%(.*), %(.*), %(.*)[)] #")
        m3 = pattern3.search(self.nodes_str)
        self.inputs.append('%' + m3.groups()[2])
        pattern4 = re.compile(r"%(.*) : Tensor = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%(.*) : Tensor = aten::t")
        m4 = pattern4.search(self.nodes_str)
        self.node_name = ['%' + m4.groups()[0]]
        
        
            
   
        
    
    