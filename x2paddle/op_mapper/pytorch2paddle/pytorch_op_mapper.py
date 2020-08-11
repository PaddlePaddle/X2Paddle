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
from x2paddle.core.program import PaddleGraph
from x2paddle.op_mapper.pytorch2paddle import prim
from x2paddle.op_mapper.pytorch2paddle import aten


class PyTorchOpMapper(OpMapper):
    def __init__(self, decoder):
        super(PyTorchOpMapper, self).__init__()
        self.script = decoder.script
        self.paddle_params = dict()
        self.outputs_info = {}  # key为output unique id，value为当前节点的输出名字
        self.pytorch_params = {}  # key为节点名，value为参数
        self.attrs = {}  # key为节点名，value为属性值
        self.output_index = 0
        self.dygraph_name_id = {}  # 动态图__init__输出名字中的id，key为kernel类型，value为id
        # 转换
        self.graph, _ = self.traverse(decoder.graph)

    def traverse(self, script_graph, control_node=None, father_layer=None):
        # 用于获取graph的输入
        def _update_graph_inputs(inputs, outputs):
            current_node_outputs.extend(outputs)
            for name in inputs:
                if name not in current_node_outputs:
                    graph_inputs.append(name)

        # 初始化
        graph = PaddleGraph(father_layer)
        current_node_outputs = []
        graph_inputs = []
        # 转换输入节点
        if isinstance(script_graph, torch._C.Graph):
            for i, ivalue in enumerate(script_graph.inputs()):
                node = ivalue.node()
                if str(ivalue.type()) != "Tensor":
                    graph.set_name(str(ivalue.type()).split(".")[-1])
                inputs, outputs = self.data(graph, node, ivalue.unique())
        # 转换中间节点
        for node in script_graph.nodes():
            kind = node.kind()
            func_name = kind.replace('::', '_')
            if hasattr(prim, func_name):
                func = getattr(prim, func_name)
                inputs, outputs = func(self, graph, node)
                _update_graph_inputs(inputs, outputs)
            elif hasattr(aten, func_name):
                func = getattr(aten, func_name)
                inputs, outputs = func(self, graph, node)
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
                script_unique_id = ivalue.unique()
                inputs, outputs = self.equal(
                    graph,
                    node,
                    uid=script_unique_id,
                    control_node=control_node,
                    index=i)
                _update_graph_inputs(inputs, outputs)
        # 设置graph的参数
        if isinstance(script_graph, torch._C.Graph):
            graph.set_parameters(self.paddle_params)
        return graph, graph_inputs

    def _get_outputs_name(self, node):
        outputs_name = []
        for output_ivalue in node.outputs():
            output_name = 'x' + str(self.output_index)
            script_unique_id = output_ivalue.unique()
            if script_unique_id in self.outputs_info:
                output_name = self.outputs_info[script_unique_id]
            self.outputs_info[script_unique_id] = output_name
            self.output_index += 1
            outputs_name.append(output_name)
        # if节点没有输出的情况
        if len(list(node.outputs())) == 0:
            output_name = 'x' + str(self.output_index)
            self.output_index += 1
            outputs_name.append(output_name)
        return outputs_name

    def _check_input(self,
                     graph,
                     node,
                     output_name,
                     node_outputs,
                     add_dim=False):
        if node.kind() == "prim::GetAttr":
            param = self.pytorch_params[output_name]
            if isinstance(param, np.ndarray):
                if add_dim:
                    param = param[np.newaxis, :]
                self.paddle_params[output_name] = param
                graph.add_layer(
                    "fluid.dygraph.base.to_variable",
                    inputs={},
                    outputs=[output_name],
                    value="params[{}]".format(string(output_name)))
            else:
                graph.add_layer(
                    "prim.constant",
                    inputs={},
                    outputs=[output_name],
                    value=string(param) if isinstance(param, str) else param)
            node_outputs.append(output_name)

    def data(self, graph, node, uid):
        for output_ivalue in node.outputs():
            script_unique_id = output_ivalue.unique()
            if script_unique_id in self.outputs_info or script_unique_id != uid:
                continue
            node_name = 'x' + str(self.output_index)
            self.outputs_info[script_unique_id] = node_name
            self.output_index += 1
        output_name = self.outputs_info[uid]
        graph.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[node_name],
            value=output_name)
        return [], [output_name]

    def equal(self, graph, node, uid=None, control_node=None, index=None):
        if control_node is not None and index is not None:
            kind = control_node.kind()
            # block的输出
            input_node_name = self.outputs_info[uid]
            control_output_id = index
            if kind == "prim::Loop":
                control_output_id = index - 1
            output_ivalue = list(control_node.outputs())[
                control_output_id].unique()
            output_node_name = self.outputs_info[output_ivalue]
            graph.add_layer(
                "prim.equal",
                inputs={'input': input_node_name},
                outputs=[output_node_name])
            return [input_node_name], [output_node_name]
