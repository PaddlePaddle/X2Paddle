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

import torch
import numpy as np
from x2paddle.core.op_mapper import OpMapper
from x2paddle.core.util import *
from x2paddle.core.paddle_graph import PaddleGraph
from x2paddle.op_mapper.pytorch2paddle import prim
from x2paddle.op_mapper.pytorch2paddle import aten


class PyTorchOpMapper(OpMapper):
    def __init__(self, decoder):
        super(PyTorchOpMapper, self).__init__()
        self.script = decoder.script
        self.paddle_params = dict()
        self.node_index = 0  # 用于构造节点的名字
        self.node_names = {}  # key为output unique id，value为当前节点的名字
        self.pytorch_params = {}  # key为节点名，value为参数
        self.attrs = {}  # key为节点名，value为属性值
        self.middle_numpy = {}
        self.dygraph_name_id = {}
        # 转换
        self.graph, _ = self.traverse(decoder.graph)

    def traverse(self, script_graph, control_node=None):
        # 用于获取graph的输入
        def _update_graph_inputs(inputs, outputs):
            current_node_outputs.extend(outputs)
            for name in inputs:
                if name not in current_node_outputs:
                    graph_inputs.append(name)

        graph = PaddleGraph()
        current_node_outputs = []
        graph_inputs = []
        # 转换输入节点
        if isinstance(script_graph, torch._C.Graph):
            for i, ivalue in enumerate(script_graph.inputs()):
                node = ivalue.node()
                if str(ivalue.type()) != "Tensor":
                    graph.set_name(str(ivalue.type()).split(".")[-1])
                outputs, inputs = self.data(graph, node, ivalue.unique())
        # 转换中间节点
        for node in script_graph.nodes():
            kind = node.kind()
            func_name = kind.replace('::', '_')
            if hasattr(prim, func_name):
                func = getattr(prim, func_name)
                outputs, inputs = func(self, graph, node)
                _update_graph_inputs(inputs, outputs)
            elif hasattr(aten, func_name):
                func = getattr(aten, func_name)
                outputs, inputs = func(self, graph, node)
                _update_graph_inputs(inputs, outputs)
            else:
                raise Exception("The kind {} in model is not supported yet.".
                                format(node.kind()))
        # 转换输出节点
        if hasattr(script_graph, 'returnNode'):
            for i, ivalue in enumerate(script_graph.returnNode().inputs()):
                if control_node.kind() == "prim::Loop" and i == 0:
                    continue
                node = ivalue.node()
                unique_id = ivalue.unique()
                outputs, inputs = self.equal(
                    graph,
                    node,
                    unique_id=unique_id,
                    control_node=control_node,
                    index=i)
                _update_graph_inputs(inputs, outputs)
        # 设置graph的参数
        if isinstance(script_graph, torch._C.Graph):
            self.paddle_params.update(self.middle_numpy)
            graph.set_parameters(self.paddle_params)
        return graph, graph_inputs

    def _get_node_name(self, node):
        node_names = []
        for output_ivalue in node.outputs():
            node_name = 'x' + str(self.node_index)
            unique_id = output_ivalue.unique()
            if unique_id in self.node_names:
                node_name = self.node_names[unique_id]
            self.node_names[unique_id] = node_name
            self.node_index += 1
            node_names.append(node_name)
        # if节点没有输出的情况
        if len(list(node.outputs())) == 0:
            node_name = 'x' + str(self.node_index)
            self.node_index += 1
            node_names.append(node_name)
        return node_names

    def _check_input(self,
                     graph,
                     node,
                     node_name,
                     node_name_list,
                     add_dim=False):

        if node.kind() == "prim::GetAttr":
            param = self.pytorch_params[node_name]
            if isinstance(param, np.ndarray):
                if add_dim:
                    param = param[np.newaxis, :]
                self.middle_numpy[node_name] = param
                graph.add_layer(
                    "fluid.dygraph.base.to_variable",
                    inputs={},
                    outputs=[node_name],
                    value="middle_numpy[{}]".format(string(node_name)))
            else:
                graph.add_layer(
                    "prim.constant",
                    inputs={},
                    outputs=[node_name],
                    value=string(param) if isinstance(param, str) else param)
            node_name_list.append(node_name)

    def data(self, graph, node, uid):
        for output_ivalue in node.outputs():
            node_name = 'x' + str(self.node_index)
            unique_id = output_ivalue.unique()
            if unique_id in self.node_names or unique_id != uid:
                continue
            self.node_names[unique_id] = node_name
            self.node_index += 1
        node_name = self.node_names[uid]
        graph.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[node_name],
            value=node_name)
        return [node_name], []

    def equal(self, graph, node, unique_id=None, control_node=None, index=None):
        if control_node is not None and index is not None:
            kind = control_node.kind()
            # block的输出
            node_name = self.node_names[unique_id]
            output_index = index
            if kind == "prim::Loop":
                output_index = index - 1
            output_node = list(control_node.outputs())[output_index].node()
            output_node_name = self._get_node_name(output_node)[0]
            graph.add_layer(
                "prim.equal",
                inputs={'input': node_name},
                outputs=[output_node_name])
            return [output_node_name], [node_name]
