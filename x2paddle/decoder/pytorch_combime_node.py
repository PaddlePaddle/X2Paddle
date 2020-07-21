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
import copy
import torch

regular_expressions = {}
regular_expressions['AdaptiveAvgPool2d'] = (
    r"(\s*)%.*: int\[\] = aten::size[(]%.*[)] # .*(\n)(\s*)%.*: bool = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: int = prim::Constant\[value=.*\][(][)](\n)(\s*)%.*: str = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: int = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: int = aten::len[(]%.*[)] # .*(\n)(\s*)%.*: int = aten::len[(]%.*[)] # .*(\n)(\s*)%.*: bool = aten::le[(]%.*, %.*[)] # .*(\n)(\s*)= prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)= prim::RaiseException[(]%.*[)] # .*(\n)(\s*)-> [(][)](\n)(\s*)block1[(][)]:(\n)(\s*)-> [(][)](\n)(\s*)%.*: int\[\] = prim::ListConstruct[(][)](\n)(\s*).*: int = aten::len[(]%.*[)] # .*(\n)(\s*)%.*: int = aten::neg[(]%.*[)] # .*(\n)(\s*)%.*: int\[\] = aten::slice[(]%.*, %.*, %.*, %.*[)] # .*(\n)(\s*)%.*: int = aten::len[(]%.*[)] # .*(\n)(\s*)%.*: int = aten::len[(]%.*[)] # .*(\n)(\s*)%.*: int\[\] = prim::ListConstruct[(]%.*, %.*[)](\n)(\s*)%.*: int = prim::min[(]%.*[)] # .*(\n)(\s*)= prim::Loop[(]%.*, %.*[)] # .*(\n)(\s*)block0[(]%.* : int[)]:(\n)(\s*)%.*: int = aten::__getitem__[(]%.*, %.*[)] # .*(\n)(\s*)%.*: int\[\] = aten::append[(]%.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)(\s*)%.*: Tensor = aten::adaptive_avg_pool2d[(]%.*, %.*[)] # .*(\n)",
    28)
regular_expressions['Dropout'] = (
    r"(\s*)%.*: bool = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: float = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: bool = prim::GetAttr\[name=\".*\"\][(]%.*[)](\n)(\s*)%.*: str = prim::Constant\[value=\".*\"\][(][)] # .*(\n)(\s*)%.*: bool = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: float = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: float = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: bool = aten::lt[(]%.*, %.*[)] # .*(\n)(\s*)%.*: bool = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)-> [(]%.*[)](\n)(\s*)block1[(][)]:(\n)(\s*)%.*: bool = aten::gt[(]%.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)(\s*)= prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)= prim::RaiseException[(]%.*[)] # .*(\n)(\s*)-> [(][)](\n)(\s*)block1[(][)]:(\n)(\s*)-> [(][)](\n)(\s*)%.*: Tensor = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: Tensor = aten::dropout_[(]%.*, %.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)(\s*)block1[(][)]:(\n)(\s*)%.*: Tensor = aten::dropout[(]%.*, %.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)",
    26)

regular_expressions['Linear'] = (
    r"(\s*)%.*: int = prim::Constant\[value=.*\][(][)](\n)(\s*)%.*: int = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: int = aten::dim[(]%.*[)] # .*(\n)(\s*)%.*: bool = aten::eq[(]%.*, %.*[)] # .*(\n)(\s*)%.*: Tensor = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: Tensor = aten::t[(]%.*[)] # .*(\n)(\s*)%.*: Tensor = aten::addmm[(]%.*, %.*, %.*, %.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)(\s*)block1[(][)]:(\n)(\s*)%.*: Tensor = aten::t[(]%.*[)] # .*(\n)(\s*)%.*: Tensor = aten::matmul[(]%.*, %.*[)] # .*(\n)(\s*)%.*: bool = prim::Constant\[value=.*\][(][)](\n)(\s*)%.*: Tensor = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: Tensor = aten::add_[(]%.*, %.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)(\s*)block1[(][)]:(\n)(\s*)-> [(]%.*[)](\n)(\s*)-> [(]%.*[)](\n)",
    20)

regular_expressions['MaxPool2d'] = (
    r"(\s*)%.*: bool = prim::Constant\[value=.*\][(][)](\n)(\s*)%.*: int\[\] = prim::If[(].*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: int\[\] = prim::ListConstruct[(][)](\n)(\s*)-> [(](.*)[)](\n)(\s*)block1[(][)]:(\n)(\s*)-> [(]%.*[)](\n)(\s*)%.*: Tensor = aten::max_pool2d[(]%.*, %.*, %.*, %.*, %.*, %.*[)] # .*(\n)",
    8)

regular_expressions['ReLU'] = (
    r"(\s*)%.*: Tensor = prim::If(.*) # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: Tensor = aten::relu_[(].*[)] # .*(\n)(\s*)-> [(](.*)[)](\n)(\s*)block1[(][)]:(\n)(\s*)%.* : Tensor = aten::relu[(].*[)] # .*(\n)(\s*)-> [(](.*)[)](\n)",
    7)

regular_expressions['Conv2d'] = (
    r"(\s*)%.*: Tensor = aten::conv2d[(]%.*, %.*, %.*, %.*, %.*, %.*, %.*[)] # .*(\n)",
    1)

regular_expressions['Flatten'] = (
    r"(\s*)%.*: Tensor = aten::flatten[(]%.*, %.*, %.*[)] # .*(\n)", 1)

regular_expressions['Max'] = (r"(\s*)%.*: Tensor = aten::max[(]%.*[)] # .*(\n)",
                              1)

regular_expressions['GreaterThan'] = (
    r"(\s*)%.*: Tensor = aten::gt[(]%.*, %.*[)] # .*(\n)(\s*)%.*: bool = aten::Bool[(]%.*[)] # .*(\n)",
    2)

regular_expressions['Invalid_BatchNorm'] = (
    r"(\s*)%.*: int = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: str = prim::Constant\[value=\"Exception\"\][(][)] # .*(\n)(\s*)%.*: int = prim::Constant\[value=4\][(][)] # .*(\n)(\s*)%.*: int = aten::dim[(]%.*[)] # .*(\n)(\s*)%.*: bool = aten::ne[(]%.*, %.*[)] # .*(\n)(\s*)= prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)= prim::RaiseException[(]%.*[)] # .*(\n)(\s*)-> [(][)](\n)(\s*)block1[(][)]:(\n)(\s*)-> [(][)](\n)(\s*)%.*: bool = prim::GetAttr\[name=\"training\"\][(]%.*[)](\n)(\s*)= prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: Tensor = prim::GetAttr\[name=\".*\"\][(]%.*[)](\n)(\s*)%.*: Tensor = aten::add[(]%.*, %.*, %.*[)] # .*(\n)(\s*)= prim::SetAttr\[name=\".*\"\][(]%.*, %.*[)](\n)(\s*)-> [(][)](\n)(\s*)block1[(][)]:(\n)(\s*)-> [(][)](\n)",
    20)

regular_expressions['BatchNorm'] = (
    r"= prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: int\[\] = aten::size[(]%.*[)] # .*(\n)(\s*)%.*: str = prim::Constant\[value=\"Exception\"\][(][)] # .*(\n)(\s*)%.*: bool = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: int = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: int = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: int = prim::Constant\[value=.*\][(][)] # .*(\n)(\s*)%.*: int = aten::__getitem__[(]%.*, %.*[)] # .*(\n)(\s*)%.*: int = aten::len[(]%.*[)] # .*(\n)(\s*)%.*: int = aten::sub[(]%.*, %.*[)] # .*(\n)(\s*)%.*: int = prim::Loop[(]%.*, %.*, %.*[)] # .*(\n)(\s*)block0[(]%.*: int, %.*: int[)]:(\n)(\s*)%.*: int = aten::add[(]%.*, %.*[)] # .*(\n)(\s*)%.*: int = aten::__getitem__[(]%.*, %.*[)] # .*(\n)(\s*)%.*: int = aten::mul[(]%.*, %.*[)] # .*(\n)(\s*)-> [(]%.*, %.*[)](\n)(\s*)%.*: bool = aten::eq[(]%.*, %.*[)] # .*(\n)(\s*)= prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)= prim::RaiseException[(]%.*[)] # .*(\n)(\s*)-> [(][)](\n)(\s*)block1[(][)]:(\n)(\s*)-> [(][)](\n)(\s*)-> [(][)](\n)(\s*)block1[(][)]:(\n)(\s*)-> [(][)](\n)(\s*)%.*: Tensor = aten::batch_norm[(]%.*, %.*, %.*, %.*, %.*, %.*, %.*, %.*, %.*[)] # .*(\n)",
    28)

regular_expressions['ReLU6'] = (
    r".*: float = prim::Constant\[value=6.\][(][)] # .*(\n)(\s*)%.*: float = prim::Constant\[value=0.\][(][)] # .*(\n)(\s*)%.*: Tensor = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.*: Tensor = aten::hardtanh_[(]%.*, %.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)(\s*)block1[(][)]:(\n)(\s*)%.*: Tensor = aten::hardtanh[(]%.*, %.*, %.*[)] # .*(\n)(\s*)-> [(]%.*[)](\n)",
    9)


class CombinedNode:
    def __init__(self, cnode_ids, kind, inputs):
        self.cnode_ids = cnode_ids
        self.kind = kind
        self.inputs = inputs

    def get_node_ids(self):
        node_ids = []
        lines = self.nodes_str.split('/n')
        for line in lines:
            pattern = re.compile(r"%.*?:")
            ms = pattern.findall(line)
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
        super(ReLUCombinedNode, self).__init__(self.cnode_ids, 'relu',
                                               self.inputs)

    def get_combined_node_info(self):
        pattern1 = re.compile(r"%(.*) : Tensor = prim::If[(](.*)[)] #")
        m1 = pattern1.search(self.nodes_str)
        self.cnode_ids = ['%' + m1.groups()[0]]
        pattern2 = re.compile(r"Tensor = aten::relu_[(]%(.*?)[)]")
        m2 = pattern2.search(self.nodes_str)
        self.inputs.append(m2.groups()[0])
        self.inputs.append(m1.groups()[1])


class AdaptiveAvgPool2dCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(AdaptiveAvgPool2dCombinedNode, self).__init__(
            self.cnode_ids, 'adaptive_avg_pool2d', self.inputs)

    def get_combined_node_info(self):
        pattern1 = re.compile(
            r"%(.*) : Tensor = aten::adaptive_avg_pool2d[(]%(.*), %(.*)[)] #")
        m1 = pattern1.search(self.nodes_str)
        self.cnode_ids = ['%' + m1.groups()[0]]
        self.inputs.append('%' + m1.groups()[1])
        pattern2 = re.compile(
            r"%(.*) : int = aten::__getitem__[(]%(.*), %(.*)[)] #")
        m2 = pattern2.search(self.nodes_str)
        self.inputs.append('%' + m2.groups()[1])


class DropoutCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(DropoutCombinedNode, self).__init__(self.cnode_ids, 'dropout',
                                                  self.inputs)

    def get_combined_node_info(self):
        pattern = re.compile(
            r"%(.*) : Tensor = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%.* : Tensor = aten::dropout_[(]%(.*), %(.*), %(.*)[)] #"
        )
        m = pattern.search(self.nodes_str)
        self.cnode_ids = ['%' + m.groups()[0]]
        self.inputs.append('%' + m.groups()[5])


class MaxPool2dCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(MaxPool2dCombinedNode, self).__init__(self.cnode_ids,
                                                    'max_pool2d', self.inputs)

    def get_combined_node_info(self):
        pattern1 = re.compile(r"bool = prim::Constant\[value=(.*)\]")
        m1 = pattern1.search(self.nodes_str)
        stride = None
        if int(m1.groups()[0]) == 0:
            pattern2 = re.compile(r"block1[(][)]:(\n)(\s*)-> [(]%(.*)[)]")
            m2 = pattern2.search(self.nodes_str)
            stride = '%' + m2.groups()[2]
        pattern3 = re.compile(
            r"%(.*) : Tensor = aten::max_pool2d[(]%(.*), %(.*), %(.*), %(.*), %(.*), %(.*)[)] #"
        )
        m3 = pattern3.search(self.nodes_str)
        self.cnode_ids = ['%' + m3.groups()[0]]
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
        super(LinearCombinedNode, self).__init__(self.cnode_ids, 'linear',
                                                 self.inputs)

    def get_combined_node_info(self):
        pattern1 = re.compile(
            r"%(.*) : Tensor = aten::matmul[(]%(.*?), %(.*?)[)] #")
        m1 = pattern1.search(self.nodes_str)
        self.inputs.append('%' + m1.groups()[1])
        pattern2 = re.compile(r"%(.*) : Tensor = aten::t[(]%(.*)[)] #")
        m2 = pattern2.search(self.nodes_str)
        self.inputs.append('%' + m2.groups()[1])
        pattern3 = re.compile(
            r"%(.*) : Tensor = aten::add_[(]%(.*), %(.*), %(.*)[)] #")
        m3 = pattern3.search(self.nodes_str)
        self.inputs.append('%' + m3.groups()[2])
        pattern4 = re.compile(
            r"%(.*) : Tensor = prim::If[(]%.*[)] # .*(\n)(\s*)block0[(][)]:(\n)(\s*)%(.*) : Tensor = aten::t"
        )
        m4 = pattern4.search(self.nodes_str)
        self.cnode_ids = ['%' + m4.groups()[0]]


class Conv2dCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(Conv2dCombinedNode, self).__init__(self.cnode_ids, 'conv2d',
                                                 self.inputs)

    def get_combined_node_info(self):
        pattern1 = re.compile(r"%(.*) : Tensor")
        m1 = pattern1.search(self.nodes_str)
        self.cnode_ids = ['%' + m1.groups()[0]]
        pattern2 = re.compile(
            r"[(]%(.*), %(.*), %(.*), %(.*), %(.*), %(.*), %(.*)[)]")
        m2 = pattern2.search(self.nodes_str)
        self.inputs.append('%' + m2.groups()[0])
        self.inputs.append('%' + m2.groups()[1])
        self.inputs.append('%' + m2.groups()[2])
        self.inputs.append('%' + m2.groups()[3])
        self.inputs.append('%' + m2.groups()[4])
        self.inputs.append('%' + m2.groups()[5])
        self.inputs.append('%' + m2.groups()[6])


class FlattenCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(FlattenCombinedNode, self).__init__(self.cnode_ids, 'flatten',
                                                  self.inputs)

    def get_combined_node_info(self):
        pattern1 = re.compile(r"%(.*) : Tensor")
        m1 = pattern1.search(self.nodes_str)
        self.cnode_ids = ['%' + m1.groups()[0]]
        pattern2 = re.compile(r"[(]%(.*), %(.*), %(.*)[)]")
        m2 = pattern2.search(self.nodes_str)
        self.inputs.append('%' + m2.groups()[0])
        self.inputs.append('%' + m2.groups()[1])
        self.inputs.append('%' + m2.groups()[2])


class MaxCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(MaxCombinedNode, self).__init__(self.cnode_ids, 'max',
                                              self.inputs)

    def get_combined_node_info(self):
        pattern = re.compile(r"%(.*) : Tensor = aten::max[(]%(.*)[)]")
        m = pattern.search(self.nodes_str)
        self.cnode_ids = ['%' + m.groups()[0]]
        self.inputs.append('%' + m.groups()[1])


class GreaterThanCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(GreaterThanCombinedNode, self).__init__(
            self.cnode_ids, 'greater_than', self.inputs)

    def get_combined_node_info(self):
        pattern1 = re.compile(r"%(.*) : bool")
        m1 = pattern1.search(self.nodes_str)
        self.cnode_ids = ['%' + m1.groups()[0]]
        pattern2 = re.compile(r"aten::gt[(]%(.*), %(.*)[)]")
        m2 = pattern2.search(self.nodes_str)
        self.inputs.append('%' + m2.groups()[0])
        self.inputs.append('%' + m2.groups()[1])


class BatchNormCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(BatchNormCombinedNode, self).__init__(self.cnode_ids, 'batchnorm',
                                                    self.inputs)

    def get_combined_node_info(self):
        pattern = re.compile(
            r"%(.*) : Tensor = aten::batch_norm[(]%(.*), %(.*), %(.*), %(.*), %(.*), %(.*), %(.*), %(.*), %(.*)[)] #"
        )
        m = pattern.search(self.nodes_str)
        self.cnode_ids = ['%' + m.groups()[0]]
        self.inputs.append('%' + m.groups()[1])
        self.inputs.append('%' + m.groups()[2])
        self.inputs.append('%' + m.groups()[3])
        self.inputs.append('%' + m.groups()[4])
        self.inputs.append('%' + m.groups()[5])
        self.inputs.append('%' + m.groups()[6])
        self.inputs.append('%' + m.groups()[7])
        self.inputs.append('%' + m.groups()[8])
        self.inputs.append('%' + m.groups()[9])


class ReLU6CombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.get_combined_node_info()
        super(ReLU6CombinedNode, self).__init__(self.cnode_ids, 'relu6',
                                                self.inputs)

    def get_combined_node_info(self):
        pattern1 = re.compile(r"%(.*) : Tensor = prim::If[(](.*)[)] #")
        m1 = pattern1.search(self.nodes_str)
        self.cnode_ids = ['%' + m1.groups()[0]]
        pattern2 = re.compile(
            r"Tensor = aten::hardtanh_[(]%(.*?), %(.*?), %(.*?)[)]")
        m2 = pattern2.search(self.nodes_str)
        self.inputs.append(m2.groups()[0])
        self.inputs.append(m1.groups()[1])


class Invalid_BatchNormCombinedNode(CombinedNode):
    def __init__(self, nodes_str):
        self.nodes_str = nodes_str
        self.inputs = []
        self.self.cnode_ids = []
        super(Invalid_BatchNormCombinedNode, self).__init__(
            self.cnode_ids, 'invalid_batchnorm', self.inputs)


def _get_str_line_index(graph_str):
    graph_str_list = graph_str.split('\n')
    line_index = {}
    for index, line in enumerate(graph_str_list):
        if line.startswith('block'):
            continue
        line_index[line] = index + 1
    return line_index


def _can_combined(combined_node, ipt_opts):
    node_ids = combined_node.get_node_ids()
    can_combined = True
    for node_id in node_ids:
        if node_id not in ipt_opts or \
                node_id in combined_node.cnode_ids:
            continue
        outputs = ipt_opts[node_id]
        if not set(outputs).issubset(node_ids):
            can_combined = False
            break
    return can_combined


def get_combined_graph(graph, ipt_opts):
    graph_str = graph.__str__()
    line_index = _get_str_line_index(graph_str)
    line_combine_infos = []
    line_combine_info = []
    used_graph_stack = []
    no_match_lines = []
    match_dict = {}
    from x2paddle.decoder import pytorch_combime_node as pcn

    def dfs(sub_graph_str, used_graph_stack):
        for op_name, regex_info in regular_expressions.items():
            regex = regex_info[0]
            line_count = regex_info[1]
            m = re.match(regex, sub_graph_str)
            current_line = sub_graph_str.split('\n')[0]
            if current_line == '':
                return
            if current_line in no_match_lines or (
                    current_line in match_dict and
                    op_name not in list(match_dict.values())):
                if 'aten' in current_line:
                    continue
                used_graph_stack.append(current_line + '\n')
                sub_graph_list = sub_graph_str.split('\n')
                sub_graph_str = '\n'.join(sub_graph_list[1:])
                if sub_graph_str == '':
                    line_combine_infos.append(copy.deepcopy(line_combine_info))
                dfs(sub_graph_str, used_graph_stack)
                out_str = used_graph_stack.pop()
                sub_graph_str = out_str + sub_graph_str
                if len(out_str.split('\n')) > 2 or 'aten' in out_str:
                    line_combine_info.pop()
                return
            if m is None:
                if 'aten' in current_line:
                    continue
                else:
                    used_graph_stack.append(current_line + '\n')
                    sub_graph_list = sub_graph_str.split('\n')
                    sub_graph_str = '\n'.join(sub_graph_list[1:])
                    l = sub_graph_str.split('\n')[0]
                    is_return = False
                    if sub_graph_str == '':
                        is_return = True
                        line_combine_infos.append(
                            copy.deepcopy(line_combine_info))
                    dfs(sub_graph_str, used_graph_stack)
                    out_str = used_graph_stack.pop()
                    sub_graph_str = out_str + sub_graph_str
                    if len(out_str.split('\n')) > 2 or 'aten' in out_str:
                        line_combine_info.pop()
                    if is_return:
                        return
                if op_name == list(regular_expressions.keys())[-1]:
                    if current_line not in list(match_dict.keys()):
                        no_match_lines.append(current_line)
            else:
                match_str = m.group()
                cnode = getattr(pcn, op_name + 'CombinedNode')(match_str)
                if not _can_combined(cnode, ipt_opts):
                    continue
                if match_str.split('\n')[0] in match_dict:
                    match_dict[match_str.split('\n')[0]].append(op_name)
                else:
                    match_dict[match_str.split('\n')[0]] = [op_name]
                used_graph_stack.append(match_str)
                sub_graph_str = sub_graph_str.replace(match_str, '')
                index = line_index[current_line]
                line_combine_info.append({index: [line_count, cnode]})
                if sub_graph_str == '':
                    line_combine_infos.append(copy.deepcopy(line_combine_info))
                dfs(sub_graph_str, used_graph_stack)
                out_str = used_graph_stack.pop()
                sub_graph_str = out_str + sub_graph_str
                if len(out_str.split('\n')) > 2 or 'aten' in out_str:
                    line_combine_info.pop()

    dfs(graph_str, used_graph_stack)
    min_line_count = len(line_index)
    origin_line_count = len(line_index)
    if len(line_combine_infos) == 0:
        raise Exception('The graph can not be combined.')
    final_line_combine_info = line_combine_infos[0]
    for info in line_combine_infos:
        for lci in info:
            origin_line_count -= (list(lci.values())[0][0] - 1)
        if origin_line_count < min_line_count:
            min_line_count = origin_line_count
            final_line_combine_info = info
    return final_line_combine_info
