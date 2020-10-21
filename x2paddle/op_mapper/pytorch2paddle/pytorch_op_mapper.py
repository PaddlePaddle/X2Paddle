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
        self.split_len = {}  # split的长度
        # 转换
        self.check_op(decoder.graph)
        self.graph, _ = self.traverse(decoder.graph)

    def check_op(self, script_graph):
        def _update_op_list(graph):
            for node in graph.nodes():
                op_list.append(node.kind())
                for block in node.blocks():
                    _update_op_list(block)

        op_list = list()
        _update_op_list(script_graph)
        op_list = list(set(op_list))
        unsupported_op_list = []
        for op in op_list:
            func_name = op.replace('::', '_')
            if not (hasattr(prim, func_name) or hasattr(aten, func_name)):
                unsupported_op_list.append(op)
        if len(unsupported_op_list) > 0:
            raise Exception("The kind {} in model is not supported yet.".format(
                unsupported_op_list))

    def traverse(self, script_graph, parent_layer=None):
        # 用于获取graph的输入
        def _update_graph_inputs(kind, inputs, outputs):
            # extend只能放更新graph_inputs之前的情况：
            # 1. loop的输出i也是输入；i是输入的原因是：子图中为父图得到的。
            # 2. 在_check_input中需要使用to_variable。
            # extend只能放更新graph_inputs之后的情况：
            # 使用了append。
            if kind != "aten::append":
                current_node_outputs.extend(outputs)
            for name in inputs:
                if name not in current_node_outputs:
                    graph_inputs.append(name)
            if kind == "aten::append":
                current_node_outputs.extend(outputs)

        # 初始化
        graph = PaddleGraph(parent_layer, graph_type="dygraph")
        if str(type(self.script))=="<class 'torch.jit.TopLevelTracedModule'>":
            graph.set_script(self.script)
        current_node_outputs = []
        graph_inputs = []
        # 转换输入节点
        if isinstance(script_graph, torch._C.Graph):
            for i, ivalue in enumerate(script_graph.inputs()):
                node = ivalue.node()
                if str(ivalue.type()) != "Tensor":
                    graph.set_name(str(ivalue.type()).split(".")[-1])
                    continue
                inputs, outputs = self.data(graph, node, ivalue.unique())
        # 转换中间节点
        for node in script_graph.nodes():
            kind = node.kind()
            func_name = kind.replace('::', '_')
            if hasattr(prim, func_name):
                func = getattr(prim, func_name)
                inputs, outputs = func(self, graph, node)
                _update_graph_inputs(kind, inputs, outputs)
            elif hasattr(aten, func_name):
                func = getattr(aten, func_name)
                inputs, outputs = func(self, graph, node)
                _update_graph_inputs(kind, inputs, outputs)

        # 转换输出节点
        if hasattr(script_graph, 'returnNode'):
            for i, ivalue in enumerate(script_graph.returnNode().inputs()):
                if parent_layer.kernel == "prim.loop" and i == 0:
                    continue
                node = ivalue.node()
                script_unique_id = ivalue.unique()
                inputs, outputs = self.equal(
                    graph,
                    node,
                    uid=script_unique_id,
                    parent_layer=parent_layer,
                    index=i)
                _update_graph_inputs("equal", inputs, outputs)

        # 设置graph的参数和输出节点
        if isinstance(script_graph, torch._C.Graph):
            graph.set_parameters(self.paddle_params)
            if hasattr(script_graph, 'return_node'):
                inputs_name, inputs_node = self._get_inputs_name(
                    script_graph.return_node())
                graph.outputs = inputs_name
        # 更新split参数
        for layer in graph.layers.values():
            if layer.kernel == "fluid.layers.split" and "num_or_sections" in layer.attrs:
                layer.attrs["num_or_sections"] = self.split_len[layer.outputs[
                    0]]
        return graph, graph_inputs

    def _get_outputs_name(self, node, attr_name=None):
        outputs_name = []
        for output_ivalue in node.outputs():
            script_unique_id = output_ivalue.unique()
            if attr_name is None:
                output_name = 'x' + str(self.output_index)
                if script_unique_id in self.outputs_info:
                    output_name = self.outputs_info[script_unique_id]
            else:
                output_name = attr_name.replace(".", "_")
            self.outputs_info[script_unique_id] = output_name
            self.output_index += 1

            outputs_name.append(output_name)
        # if或loop节点没有输出的情况
        if len(list(node.outputs())) == 0:
            output_name = '_x' + str(self.output_index)
            self.output_index += 1
            outputs_name.append(output_name)
        return outputs_name

    def _check_input(self,
                     graph,
                     node,
                     output_name,
                     node_outputs,
                     scope_name,
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
                    scope_name=scope_name,
                    value="params[{}]".format(string(output_name)))
            else:
                if isinstance(param, dict) and "Tensor" in param and \
                "parent_layer_id" in param:
                    if graph.parent_layer is not None:
                        # 当某个param被2个控制流（if-else）赋值时，else不可以引用if中的赋值结果
                        id1 = param["parent_layer_id"]
                        id2 = graph.parent_layer.id
                        id1_part = id1.split(".")
                        id2_part = id2.split(".")
                        if len(id1_part) >= len(id2_part):
                            for i in range(len(id1_part)):
                                if id1_part[i] == id2_part[i]:
                                    continue
                                else:
                                    if id1_part[i] == "0" and id2_part[
                                            i] == "1":
                                        if add_dim:
                                            param = param[np.newaxis, :]
                                        self.paddle_params[output_name] = param
                                        graph.add_layer(
                                            "fluid.dygraph.base.to_variable",
                                            inputs={},
                                            outputs=[output_name],
                                            scope_name=scope_name,
                                            value="params[{}]".format(
                                                string(output_name)))
                                        node_outputs.append(output_name)
                                        return
                    # 若if-else外，则可直接引用if-else中的赋值结果
                    graph.add_layer(
                        "prim.constant",
                        inputs={},
                        outputs=[output_name],
                        scope_name=scope_name,
                        value=param["Tensor"])
                else:
                    graph.add_layer(
                        "prim.constant",
                        inputs={},
                        outputs=[output_name],
                        scope_name=scope_name,
                        value=string(param)
                        if isinstance(param, str) else param)
            node_outputs.append(output_name)
        elif node.kind() == "prim::Constant" and output_name in self.pytorch_params:
            param = self.pytorch_params[output_name]
            self.paddle_params[output_name] = param
            graph.add_layer(
                "fluid.dygraph.base.to_variable",
                inputs={},
                outputs=[output_name],
                scope_name=scope_name,
                value="params[{}]".format(string(output_name)))     

            
    def _get_inputs_name(self, node):
        inputs_name = []
        inputs_node = []
        for script_input_ivalue in node.inputs():
            script_input_node = script_input_ivalue.node()
            script_input_unique_id = script_input_ivalue.unique()
            input_name = self.outputs_info[script_input_unique_id]
            inputs_node.append(script_input_node)
            inputs_name.append(input_name)
        return inputs_name, inputs_node
    

    def data(self, graph, node, uid):
        scope_name = self.normalize_scope_name(node)
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
            scope_name=scope_name,
            value=output_name)
        return [], [output_name]

    def equal(self, graph, node, uid=None, parent_layer=None, index=None):
        scope_name = self.normalize_scope_name(node)
        if parent_layer is not None and index is not None:
            # block的输出
            input_node_name = self.outputs_info[uid]
            control_output_id = index
            if parent_layer.kernel == "prim.loop":
                control_output_id = index - 1
            output_node_name = parent_layer.outputs[control_output_id]
            current_outputs = [output_node_name]
            self._check_input(graph, node, input_node_name, current_outputs)
            graph.add_layer(
                "prim.equal",
                inputs={'input': input_node_name},
                outputs=[output_node_name],
                scope_name=scope_name)
            return [input_node_name], current_outputs

    def normalize_scope_name(self, node):
        """ 对scope的名字进行标准化。
        """
        scope_name = node.scopeName()
        if scope_name == "":
            return scope_name
        scope_name_part = scope_name.split("/")
        for index in range(len(scope_name_part) - 1):
            if scope_name_part[index] in scope_name_part[index + 1]:
                continue
            last_name_segments = scope_name_part[index].split(".")
            name_segments = scope_name_part[index + 1].split(".")
            for j, name in enumerate(last_name_segments):
                name_segments[j] = name
            scope_name_part[index + 1] = ".".join(name_segments)
        last_name = scope_name_part[-1]
        name_segments = last_name.split(".")
        return "/".join(name_segments[1:])
                