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

import functools
import numbers
import numpy as np
import math
import torch
import paddle.fluid as fluid
from x2paddle.core.op_mapper import OpMapper
from x2paddle.core.util import *
from x2paddle.core.program import PaddleProgram


class PyTorchOpMapper(OpMapper):
    def __init__(self, decoder):
        super(PyTorchOpMapper, self).__init__()
        self.script = decoder.script
        self.paddle_params = dict()
        self.node_index = 0  # 用于构造节点的名字
        self.output_nodenames = {}  # key为output unique id，value为当前节点的名字
        self.pytorch_params = {}  # key为节点名，value为参数
        self.attrs = {}  # key为节点名，value为属性值
        self.middle_numpy = {}
        # 转换
        self.prog, _ = self.parser(decoder.graph)

    def parser(self, graph, control_node=None):
        # 用于获取program的输入
        def _update_prog_inputs(node_name_list, inputs):
            current_node_name_list.extend(node_name_list)
            for name in inputs:
                if name not in current_node_name_list:
                    prog_inputs.append(name)

        prog = PaddleProgram()
        current_node_name_list = []
        prog_inputs = []
        # 转换输入节点
        if isinstance(graph, torch._C.Graph):
            for i, ivalue in enumerate(graph.inputs()):
                node = ivalue.node()
                node_name_list, inputs = self.data(prog, node, ivalue.unique())
        # 转换中间节点
        for node in graph.nodes():
            kind = node.kind()
            op = kind.replace('::', '_')
            if hasattr(self, op):
                func = getattr(self, op)
                node_name_list, inputs = func(prog, node)
                _update_prog_inputs(node_name_list, inputs)
            else:
                raise Exception("The kind {} in model is not supported yet.".
                                format(node.kind()))

        # 转换输出节点
        if hasattr(graph, 'returnNode'):
            for i, ivalue in enumerate(graph.returnNode().inputs()):
                if control_node.kind() == "prim::Loop" and i == 0:
                    continue
                node = ivalue.node()
                unique_id = ivalue.unique()
                node_name_list, inputs = self.equal(
                    prog,
                    node,
                    unique_id=unique_id,
                    control_node=control_node,
                    index=i)
                _update_prog_inputs(node_name_list, inputs)
        return prog, prog_inputs

    def _get_node_name(self, node):
        node_names = []
        for output_ivalue in node.outputs():
            node_name = 'x' + str(self.node_index)
            unique_id = output_ivalue.unique()
            if unique_id in self.output_nodenames:
                node_name = self.output_nodenames[unique_id]
            self.output_nodenames[unique_id] = node_name
            self.node_index += 1
            node_names.append(node_name)
        if len(list(node.outputs())) == 0:
            node_name = 'x' + str(self.node_index)
            self.node_index += 1
            node_names.append(node_name)
        return node_names

    def _check_input(self, prog, node, node_name, node_name_list,
                     add_dim=False):
        if node.kind() == "prim::GetAttr":
            param = self.pytorch_params[node_name]
            if isinstance(param, np.ndarray):
                if add_dim:
                    param = param[np.newaxis, :]
                self.middle_numpy[node_name] = param
                prog.add_layer(
                    node_name,
                    "fluid.dygraph.base.to_variable",
                    inputs={},
                    outputs=[node_name],
                    value="middle_numpy[{}]".format(string(node_name)))
            else:
                prog.add_layer(
                    node_name,
                    "prim.constant",
                    inputs={},
                    outputs=[node_name],
                    value=string(param) if isinstance(param, str) else param)
            node_name_list.append(node_name)

    def data(self, prog, node, uid):
        for output_ivalue in node.outputs():
            node_name = 'x' + str(self.node_index)
            unique_id = output_ivalue.unique()
            if unique_id in self.output_nodenames or unique_id != uid:
                continue
            self.output_nodenames[unique_id] = node_name
            self.node_index += 1
        node_name = self.output_nodenames[uid]
        prog.add_layer(
            node_name,
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[node_name],
            value=node_name)
        return [node_name], []

    def equal(self, prog, node, unique_id=None, control_node=None, index=None):
        if control_node is not None and index is not None:
            kind = control_node.kind()
            # block的输出
            node_name = self.output_nodenames[unique_id]
            output_index = index
            if kind == "prim::Loop":
                output_index = index - 1
            output_node = list(control_node.outputs())[output_index].node()
            output_node_name = self._get_node_name(output_node)[0]
            prog.add_layer(
                node_name,
                "prim.equal",
                inputs={'input': node_name},
                outputs=[output_node_name])
            return [output_node_name], [node_name]

    def prim_GetAttr(self, prog, node):
        node_name = self._get_node_name(node)[0]
        field_name_list = [node.s('name')]
        while True:
            input_node = list(node.inputs())[0].node()
            try:
                field_name_list.insert(0, input_node.s('name'))
                node = input_node
            except Exception:
                break
        part_script = self.script
        for field_name in field_name_list:
            if hasattr(part_script, field_name):
                param = getattr(part_script, field_name)
                if isinstance(param, torch.nn.parameter.Parameter):
                    param = param.detach().numpy()
                self.pytorch_params[node_name] = param
                part_script = param
        return [node_name], []

    def prim_Constant(self, prog, node):
        node_name = self._get_node_name(node)[0]
        output = list(node.outputs())[0]
        value = output.toIValue()
        self.attrs[node_name] = value
        if isinstance(value, str):
            value = string(value)
        prog.add_layer(
            node_name,
            "prim.constant",
            inputs={},
            outputs=[node_name],
            value=value)
        return [node_name], []

    def prim_ListConstruct(self, prog, node):
        node_name = self._get_node_name(node)[0]
        inputs = {}
        for i, input_ivalue in enumerate(node.inputs()):
            input_node = input_ivalue.node()
            input_unique_id = input_ivalue.unique()
            input_node_name = self.output_nodenames[input_unique_id]
            inputs['input{}'.format(i)] = input_node_name
        prog.add_layer(
            node_name, "prim.list", inputs=inputs, outputs=[node_name])
        return [node_name], list(inputs.values())

    def prim_RaiseException(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(prog, input_node, input_node_name, node_name_list)
        prog.add_layer(
            node_name,
            "prim.exception",
            inputs={'input': input_node_name},
            outputs=[node_name])
        return node_name_list, [input_node_name]

    def prim_Loop(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        loop_inputs = {}
        block = list(node.blocks())[0]
        loop_outputs = [node_name]
        for i, block_input_ivalue in enumerate(block.inputs()):
            block_input_node_name = 'x' + str(self.node_index)
            unique_id = block_input_ivalue.unique()
            if unique_id not in self.output_nodenames:
                self.output_nodenames[unique_id] = block_input_node_name
                self.node_index += 1
            if i == 0:
                loop_input_node = list(node.inputs())[0].node()
                loop_input_unique_id = list(node.inputs())[0].unique()
                loop_input_node_name = self.output_nodenames[
                    loop_input_unique_id]
                self._check_input(prog, loop_input_node, loop_input_node_name,
                                  node_name_list)
                loop_inputs['input'] = loop_input_node_name
                loop_outputs.append(block_input_node_name)
                node_name_list.append(block_input_node_name)
            else:
                loop_input_node = list(node.inputs())[i + 1].node()
                loop_input_unique_id = list(node.inputs())[i + 1].unique()
                loop_input_node_name = self.output_nodenames[
                    loop_input_unique_id]
                self._check_input(prog, loop_input_node, loop_input_node_name,
                                  node_name_list)
                prog.add_layer(
                    node_name,
                    "prim.equal",
                    inputs={'input': loop_input_node_name},
                    outputs=[block_input_node_name])
                node_name_list.append(block_input_node_name)
        prog.add_layer(
            node_name, "prim.loop", inputs=loop_inputs, outputs=loop_outputs)
        current_layer = prog.layers[-1]
        block_prog, prog_inputs = self.parser(block, node)
        for i, input_name in enumerate(prog_inputs):
            if input_name == loop_outputs[1]:
                continue
            current_layer.inputs['input-{}'.format(i)] = input_name
        current_layer.add_block(block_prog)
        return node_name_list, list(current_layer.inputs.values())

    def prim_If(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(prog, input_node, input_node_name, node_name_list)
        prog.add_layer(node_name, "prim.if", {'input': input_node_name},
                       [node_name])
        current_layer = prog.layers[-1]
        block0 = list(node.blocks())[0]
        block0_prog, prog_inputs0 = self.parser(block0, node)
        len0 = 0
        for i, input_name in enumerate(prog_inputs0):
            current_layer.inputs['input-{}'.format(i)] = input_name
            len0 = i
        current_layer.add_block(block0_prog)
        block1 = list(node.blocks())[1]
        block1_prog, prog_inputs1 = self.parser(block1, node)
        for i, input_name in enumerate(prog_inputs1):
            current_layer.inputs['input-{}'.format(len0 + 1 + i)] = input_name
        current_layer.add_block(block1_prog)
        return node_name_list, list(current_layer.inputs.values())

    def prim_min(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(prog, input_node, input_node_name, node_name_list)
        prog.add_layer(
            node_name,
            "prim.min",
            inputs={'input': input_node_name},
            outputs=[node_name])
        return node_name_list, [input_node_name]

    def aten_adaptive_avg_pool2d(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        adapoo2d_inputs = []
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(prog, input_node, input_node_name, node_name_list)
        adapoo2d_inputs.append(input_node_name)
        attr_node = list(node.inputs())[1].node()
        attr_unique_id = list(node.inputs())[1].unique()
        attr_node_name = self.output_nodenames[attr_unique_id]
        attrs = {}
        attrs["pool_size"] = self.attrs[
            attr_node_name] if attr_node_name in self.attrs else attr_node_name
        if attr_node_name not in self.attrs:
            adapoo2d_inputs.append(attr_node_name)
        attrs["pool_type"] = string("avg")
        prog.add_layer(
            node_name,
            "fluid.layers.adaptive_pool2d",
            inputs={"input": input_node_name},
            outputs=[node_name],
            **attrs)
        return node_name_list, [input_node_name]

    def aten_addmm(self, prog, node):
        node_name = self._get_node_name(node)[0]
        inputs = {}
        attrs = {}
        addmm_inputs = []
        node_name_list = [node_name]
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(
            prog, input_node, input_node_name, node_name_list, add_dim=True)
        inputs['input'] = input_node_name
        addmm_inputs.append(input_node_name)
        x_node = list(node.inputs())[1].node()
        x_unique_id = list(node.inputs())[1].unique()
        x_node_name = self.output_nodenames[x_unique_id]
        self._check_input(prog, x_node, x_node_name, node_name_list)
        inputs['x'] = x_node_name
        addmm_inputs.append(x_node_name)
        y_node = list(node.inputs())[2].node()
        y_unique_id = list(node.inputs())[2].unique()
        y_node_name = self.output_nodenames[y_unique_id]
        self._check_input(prog, y_node, y_node_name, node_name_list)
        inputs['y'] = y_node_name
        addmm_inputs.append(y_node_name)
        beta_node = list(node.inputs())[3].node()
        beta_unique_id = list(node.inputs())[3].unique()
        beta_node_name = self.output_nodenames[beta_unique_id]
        attrs['beta'] = self.attrs[
            beta_node_name] if beta_node_name in self.attrs else beta_node_name
        if beta_node_name not in self.attrs:
            addmm_inputs.append(beta_node_name)
        alpha_node = list(node.inputs())[4].node()
        alpha_unique_id = list(node.inputs())[4].unique()
        alpha_node_name = self.output_nodenames[alpha_unique_id]
        attrs['alpha'] = self.attrs[
            alpha_node_name] if alpha_node_name in self.attrs else alpha_node_name
        if alpha_node_name not in self.attrs:
            addmm_inputs.append(alpha_node_name)
        prog.add_layer(
            node_name,
            "fluid.layers.addmm",
            inputs=inputs,
            outputs=[node_name],
            **attrs)
        return node_name_list, addmm_inputs

    def aten_add_(self, prog, node):
        node_name = self._get_node_name(node)[0]
        inputs = {}
        attrs = {}
        add_inputs = []
        node_name_list = [node_name]
        x_node = list(node.inputs())[0].node()
        x_unique_id = list(node.inputs())[0].unique()
        x_node_name = self.output_nodenames[x_unique_id]
        self._check_input(prog, x_node, x_node_name, node_name_list)
        inputs['x'] = x_node_name
        add_inputs.append(x_node_name)
        y_node = list(node.inputs())[1].node()
        y_unique_id = list(node.inputs())[1].unique()
        y_node_name = self.output_nodenames[y_unique_id]
        self._check_input(
            prog, y_node, y_node_name, node_name_list, add_dim=True)
        inputs['y'] = y_node_name
        add_inputs.append(y_node_name)
        alpha_node = list(node.inputs())[2].node()
        alpha_unique_id = list(node.inputs())[2].unique()
        alpha_node_name = self.output_nodenames[alpha_unique_id]
        attrs['alpha'] = self.attrs[
            alpha_node_name] if alpha_node_name in self.attrs else alpha_node_name
        if alpha_node_name not in self.attrs:
            add_inputs.append(alpha_node_name)
        # out = x + alpha * y
        prog.add_layer(
            node_name, "prim.add", inputs=inputs, outputs=[node_name], **attrs)
        return node_name_list, add_inputs

    def aten_append(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        inputs = {}
        for i, input_ivalue in enumerate(node.inputs()):
            input_node = input_ivalue.node()
            input_unique_id = input_ivalue.unique()
            input_node_name = self.output_nodenames[input_unique_id]
            self._check_input(prog, input_node, input_node_name, node_name_list)
            if i == 0:
                inputs['list'] = input_node_name
            else:
                inputs['element'] = input_node_name
        prog.add_layer(
            node_name, "prim.append", inputs=inputs, outputs=[node_name])
        return node_name_list, list(inputs.values())

    def aten_conv2d(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        inputs = {}
        attrs = {}
        conv2d_inputs = []
        for i, input_ivalue in enumerate(node.inputs()):
            input_node = input_ivalue.node()
            input_unique_id = input_ivalue.unique()
            input_node_name = self.output_nodenames[input_unique_id]
            if i == 0:
                inputs['input'] = input_node_name
                conv2d_inputs.append(input_node_name)
            elif i == 1:
                weights_name = input_node_name
                weights = self.pytorch_params[input_node_name]
                self.paddle_params["conv2d_" + node_name + '.weight'] = weights
                attrs['num_filters'] = weights.shape[0]
                attrs['filter_size'] = weights.shape[2:]
            elif i == 2:
                if input_node_name in self.pytorch_params:
                    bias = self.pytorch_params[input_node_name]
                    self.paddle_params["conv2d_" + node_name + '.bias'] = bias
                else:
                    self.paddle_params["conv2d_" + node_name + '.bias'] = False
            elif i == 3:
                attrs['stride'] = self.attrs[input_node_name]
            elif i == 4:
                attrs['padding'] = self.attrs[input_node_name]
            elif i == 5:
                attrs['dilation'] = self.attrs[input_node_name]
            elif i == 6:
                if input_node_name in self.attrs:
                    attrs['groups'] = self.attrs[input_node_name]
                    attrs['num_channels'] = weights.shape[1] * self.attrs[
                        input_node_name]
                else:
                    attrs['groups'] = input_node_name
                    attrs[
                        'num_channels'] = weights_name + '.shape[1] * ' + input_node_name
                    conv2d_inputs.append(input_node_name)
                    conv2d_inputs.append(weights_name)
        prog.add_layer(
            "conv2d_" + node_name,
            "fluid.dygraph.Conv2D",
            inputs=inputs,
            outputs=[node_name],
            **attrs)
        return node_name_list, conv2d_inputs

    def aten_dim(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(prog, input_node, input_node_name, node_name_list)
        prog.add_layer(
            node_name,
            "prim.shape",
            inputs={'input': input_node_name},
            outputs=[node_name])
        prog.add_layer(
            node_name,
            "prim.len",
            inputs={'input': node_name},
            outputs=[node_name])
        return node_name_list, [input_node_name]

    def aten_dropout(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(prog, input_node, input_node_name, node_name_list)
        prog.add_layer(
            'dropout_' + node_name,
            "fluid.dygraph.Dropout",
            inputs={"input": input_node_name},
            outputs=[node_name],
            p=0.0)
        return node_name_list, [input_node_name]

    def aten_eq(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        inputs = {}
        eq_inputs = []
        for i, input_ivalue in enumerate(node.inputs()):
            input_node = input_ivalue.node()
            input_unique_id = input_ivalue.unique()
            input_node_name = self.output_nodenames[input_unique_id]
            self._check_input(prog, input_node, input_node_name, node_name_list)
            inputs['eq{}'.format(i)] = input_node_name
            eq_inputs.append(input_node_name)
        prog.add_layer(node_name, "prim.eq", inputs=inputs, outputs=[node_name])
        return node_name_list, list(inputs.values())

    def aten_flatten(self, prog, node):
        # 目前只支持第一维的flatten
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        flatten_inputs = []
        for i, input_ivalue in enumerate(node.inputs()):
            if i == 0:
                continue
            input_node = input_ivalue.node()
            input_unique_id = input_ivalue.unique()
            input_node_name = self.output_nodenames[input_unique_id]
            self._check_input(prog, input_node, input_node_name, node_name_list)
            prog.add_layer(
                "assert_" + node_name,
                "prim.assert",
                inputs={'input': input_node_name},
                outputs=[node_name + '_assert'],
                type='eq',
                value=1 if i == 1 else -1)
            flatten_inputs.append(input_node_name)
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(prog, input_node, input_node_name, node_name_list)
        prog.add_layer(
            node_name,
            "fluid.layers.flatten",
            inputs={'x': input_node_name},
            outputs=[node_name],
            axis=1)
        flatten_inputs.append(input_node_name)
        return node_name_list, flatten_inputs

    def aten___getitem__(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        inputs = {}
        for i, input_ivalue in enumerate(node.inputs()):
            input_node = input_ivalue.node()
            input_unique_id = input_ivalue.unique()
            input_node_name = self.output_nodenames[input_unique_id]
            self._check_input(prog, input_node, input_node_name, node_name_list)
            if i == 0:
                inputs['list'] = input_node_name
            else:
                inputs['index'] = input_node_name
        prog.add_layer(
            node_name, "prim.getitem", inputs=inputs, outputs=[node_name])
        return node_name_list, list(inputs.values())

    def aten_le(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        inputs = {}
        for i, input_ivalue in enumerate(node.inputs()):
            input_node = input_ivalue.node()
            input_unique_id = input_ivalue.unique()
            input_node_name = self.output_nodenames[input_unique_id]
            self._check_input(prog, input_node, input_node_name, node_name_list)
            inputs['input{}'.format(i)] = input_node_name
        prog.add_layer(node_name, "prim.le", inputs=inputs, outputs=[node_name])
        return node_name_list, list(inputs.values())

    def aten_len(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(prog, input_node, input_node_name, node_name_list)
        prog.add_layer(
            node_name,
            "prim.len",
            inputs={'input': input_node_name},
            outputs=[node_name])
        return node_name_list, [input_node_name]

    def aten_max_pool2d(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        inputs = {}
        attrs = {}
        pool_inputs = []
        for i, input_ivalue in enumerate(node.inputs()):
            input_node = input_ivalue.node()
            input_unique_id = input_ivalue.unique()
            input_node_name = self.output_nodenames[input_unique_id]
            if i == 0:
                self._check_input(prog, input_node, input_node_name,
                                  node_name_list)
                inputs['input'] = input_node_name
                pool_inputs.append(input_node_name)
            elif i == 1:
                attrs['pool_size'] = self.attrs[input_node_name]
            elif i == 2:
                attrs['pool_stride'] = self.attrs[input_node_name]
            elif i == 3:
                attrs['pool_padding'] = self.attrs[input_node_name]
            elif i == 4:
                prog.add_layer(
                    node_name + '_assert',
                    "prim.assert",
                    inputs={'input': input_node_name},
                    outputs=[node_name + '_assert'],
                    type='eq',
                    value=[1, [1, 1]])
                pool_inputs.append(input_node_name)
            elif i == 5:
                attrs['ceil_mode'] = self.attrs[
                    input_node_name] if input_node_name in self.attrs else input_node_name
                if input_node_name not in self.attrs:
                    pool_inputs.append(input_node_name)
        attrs['pool_type'] = string('max')
        prog.add_layer(
            "max_pool2d_" + node_name,
            "fluid.dygraph.Pool2D",
            inputs=inputs,
            outputs=[node_name],
            **attrs)
        return node_name_list, pool_inputs

    def aten_matmul(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        inputs = {}
        x_node = list(node.inputs())[0].node()
        x_unique_id = list(node.inputs())[0].unique()
        x_node_name = self.output_nodenames[x_unique_id]
        self._check_input(prog, x_node, x_node_name, node_name_list)
        inputs['x'] = x_node_name
        y_node = list(node.inputs())[1].node()
        y_unique_id = list(node.inputs())[1].unique()
        y_node_name = self.output_nodenames[y_unique_id]
        inputs['y'] = y_node_name
        self._check_input(prog, y_node, y_node_name, node_name_list)
        prog.add_layer(
            node_name,
            "fluid.layers.matmul",
            inputs=inputs,
            outputs=[node_name])
        return node_name_list, list(inputs.values())

    def aten_relu_(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(prog, input_node, input_node_name, node_name_list)
        # inplace这个参数在paddle中未实现
        prog.add_layer(
            node_name,
            "fluid.layers.relu",
            inputs={"x": input_node_name},
            outputs=[node_name])
        return node_name_list, [input_node_name]

    def aten_relu6(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(prog, input_node, input_node_name, node_name_list)
        # inplace这个参数在paddle中未实现
        prog.add_layer(
            node_name,
            "fluid.layers.relu6",
            inputs={"x": input_node_name},
            outputs=[node_name],
            threshold=6.0)
        return node_name_list, [input_node_name]

    def aten_size(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(prog, input_node, input_node_name, node_name_list)
        prog.add_layer(
            node_name,
            "prim.shape",
            inputs={'input': input_node_name},
            outputs=[node_name])
        return node_name_list, [input_node_name]

    def aten_slice(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        attrs = {}
        slice_inputs = []
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        slice_inputs.append(input_node_name)
        strat_node = list(node.inputs())[1].node()
        start_unique_id = list(node.inputs())[1].unique()
        start_node_name = self.output_nodenames[start_unique_id]
        slice_inputs.append(start_node_name)
        attrs['start'] = self.attrs[
            start_node_name] if start_node_name in self.attrs else start_node_name
        if start_node_name not in self.attrs:
            self._check_input(prog, strat_node, start_node_name, node_name_list)
            slice_inputs.append(input_node_name)
        end_node = list(node.inputs())[2].node()
        end_unique_id = list(node.inputs())[2].unique()
        end_node_name = self.output_nodenames[end_unique_id]
        slice_inputs.append(end_node_name)
        attrs['end'] = self.attrs[
            end_node_name] if end_node_name in self.attrs else end_node_name
        if end_node_name not in self.attrs:
            self._check_input(prog, end_node, end_node_name, node_name_list)
            slice_inputs.append(end_node_name)
        step_node = list(node.inputs())[3].node()
        step_unique_id = list(node.inputs())[3].unique()
        step_node_name = self.output_nodenames[step_unique_id]
        slice_inputs.append(step_node_name)
        attrs['step'] = self.attrs[
            step_node_name] if step_node_name in self.attrs else step_node_name
        if step_node_name not in self.attrs:
            self._check_input(prog, step_node, step_node_name, node_name_list)
            slice_inputs.append(step_node_name)
        prog.add_layer(
            node_name,
            "prim.slice",
            inputs={'input': input_node_name},
            outputs=[node_name],
            **attrs)
        return node_name_list, [input_node_name]

    def aten_t(self, prog, node):
        node_name = self._get_node_name(node)[0]
        node_name_list = [node_name]
        input_node = list(node.inputs())[0].node()
        input_unique_id = list(node.inputs())[0].unique()
        input_node_name = self.output_nodenames[input_unique_id]
        self._check_input(prog, input_node, input_node_name, node_name_list)
        prog.add_layer(
            node_name,
            "fluid.layers.transpose",
            inputs={"x": input_node_name},
            outputs=[node_name],
            perm=[1, 0])
        return node_name_list, [input_node_name]
