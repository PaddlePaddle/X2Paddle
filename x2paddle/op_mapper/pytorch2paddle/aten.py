# -*- coding:UTF-8 -*-
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

import copy
import numpy as np
from x2paddle.core.util import name_generator, string
from x2paddle.utils import paddle_dtypes
from x2paddle.core.program import PaddleGraph

dtype_dict = {
    0: string("uint8"),
    1: string("int8"),
    2: string("int16"),
    3: string("int32"),
    4: string("int64"),
    5: string("float16"),
    6: string("float32"),
    7: string("float64"),
    11: string("bool")
}


def aten_sum(mapper, graph, node):
    """ 构造获取元素求和的paddlelayer。
    TorchScript示例:
        %x_gap.15 : Tensor =  aten::sum(%x.58, %2166, %1450, %1453)
        参数含义:
        %x_gap.15 (Tensor): 求和后的Tensor。
        %n.3 (Tensor): 求和前的Tensor。
        %2166：axis
        %1450：keepdim
        %1453：dtype
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%n.3
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    if inputs_name[2] in mapper.attrs:
        layer_attrs["keepdim"] = mapper.attrs[inputs_name[2]]
    if inputs_name[3] in mapper.attrs:
        layer_attrs["dtype"] = mapper.attrs[inputs_name[3]]
    graph.add_layer(
        "paddle.sum",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs

def aten_abs(mapper, graph, node):
    """ 构造获取绝对值的PaddleLayer。
    TorchScript示例:
        %n0.3 : Tensor = aten::abs(%n.3)
        参数含义:
        %n0.3 (Tensor): 绝对值后的Tensor。
        %n.3 (Tensor): 绝对值前的Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%n.3
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.abs",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_adaptive_avg_pool1d(mapper, graph, node):
    """ 构造average adaptive pool1d的PaddleLayer。
    TorchScript示例:
        %x.5 : Tensor = aten::adaptive_avg_pool1d(%x.3, %_output_size.1)
        参数含义:
        %x.5 (Tensor): 池化后结果Tensor。
        %x.3 (Tensor): 输入Tensor。
        %_output_size.1 (list): 自适应池化后的Tensor的长度大小。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("pool1d", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.3
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%_output_size.1
    if inputs_name[1] in mapper.attrs:
        layer_attrs["output_size"] = mapper.attrs[inputs_name[1]][0]
        graph.add_layer(
            "paddle.nn.AdaptiveAvgPool1D",
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name,
            **layer_attrs)
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["output_size"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
        graph.add_layer(
            "prim.getitem",
            inputs={"list": layer_inputs["output_size"]},
            outputs=[layer_inputs["output_size"]],
            scope_name=scope_name,
            index=0)
        graph.add_layer(
            "paddle.nn.functional.adaptive_avg_pool1d",
            inputs=layer_inputs,
            outputs=layer_outputs[1:],
            scope_name=scope_name,
            **layer_attrs)
    return current_inputs, current_outputs


def aten_adaptive_avg_pool2d(mapper, graph, node):
    """ 构造average adaptive pool2d的PaddleLayer。
    TorchScript示例:
        %x.5 : Tensor = aten::adaptive_avg_pool2d(%x.3, %_output_size.1)
        参数含义:
        %x.5 (Tensor): 池化后结果Tensor。
        %x.3 (Tensor): 输入Tensor。
        %_output_size.1 (list): 自适应池化后的Tensor的宽、高大小。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("pool2d", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.3
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%_output_size.1
    if inputs_name[1] in mapper.attrs:
        layer_attrs["output_size"] = mapper.attrs[inputs_name[1]]
        graph.add_layer(
            "paddle.nn.AdaptiveAvgPool2D",
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name,
            **layer_attrs)
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["output_size"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
        graph.add_layer(
            "paddle.nn.functional.adaptive_avg_pool2d",
            inputs=layer_inputs,
            outputs=layer_outputs[1:],
            scope_name=scope_name,
            **layer_attrs)
    return current_inputs, current_outputs


def aten_addmm(mapper, graph, node):
    """ 构造addmm的PaddleLayer，该节点实现out = alpha ∗ x ∗ y + beta ∗ input。
    TorchScript示例:
        %ret.2 : Tensor = aten::addmm(%150, %input.3, %156, %151, %152)
        参数含义:
        %ret.2 (Tensor): addmm结果Tensor。
        %150 (Tensor): 输入Tensor input。
        %input.3 (Tensor): 输入Tensor x。
        %156 (Tensor): 输入Tensor y。
        %151 (int/float): 输入alpha。
        %152 (int/float): 输入beta。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%150
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 处理输入1，即%input.3
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[1]
    # 处理输入2，即%156
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入3，即%152
    if inputs_name[3] in mapper.attrs:
        layer_attrs["beta"] = mapper.attrs[inputs_name[3]]
    else:
        mapper._check_input(graph, inputs_node[3], inputs_name[3],
                            current_outputs, scope_name)
        layer_inputs["beta"] = inputs_name[3]
        current_inputs.append(inputs_name[3])
    # 处理输入4，即%151
    if inputs_name[4] in mapper.attrs:
        layer_attrs["alpha"] = mapper.attrs[inputs_name[4]]
    else:
        mapper._check_input(graph, inputs_node[4], inputs_name[4],
                            current_outputs, scope_name)
        layer_inputs["alpha"] = inputs_name[4]
        current_inputs.append(inputs_name[4])

    graph.add_layer(
        "paddle.addmm",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_add(mapper, graph, node):
    """ 构造数值相加的PaddleLayer，该节点实现out = x + y。
    TorchScript示例:
        %296 : int = aten::add(%i.12, %288)
        参数含义:
        %296 (-): 相加结果。
        %i.12 (-): 输入数值 x。
        %288 (-): 输入数值 y。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%i.12
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%288
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.add",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_add_(mapper, graph, node):
    """ 构造数值相加的PaddleLayer，该节点实现out = x + alpha * y。
    TorchScript示例:
        %137 : Tensor = aten::add(%136, %130, %130)
        参数含义:
        %output.5 (Tensor): add结果Tensor。
        %output.2 (Tensor): 输入Tensor x。
        %150 (Tensor): 输入Tensor y。
        %151 (int/float): 输入alpha。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%output.2
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%150
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入2，即%151
    if inputs_name[2] in mapper.attrs:
        layer_attrs["alpha"] = mapper.attrs[inputs_name[2]]
    else:
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs, scope_name)
        layer_inputs["alpha"] = inputs_name[2]
        current_inputs.append(inputs_name[2])

    graph.add_layer(
        "prim.add_",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten___and__(mapper, graph, node):
    """ 构造与计算的PaddleLayer。
    TorchScript示例:
        %361 : bool = aten::__and__(%360, %358)
        参数含义:
        %361 (bool): 输出，与计算结果。
        %360 (-): 输入 x。
        %358 (-): 输入 y。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%i.12
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%288
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.and",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_append(mapper, graph, node):
    """ 构造对list进行append的PaddleLayer。
    TorchScript示例:
        %90 : int[] = aten::append(%_output_size.1, %v.1)
        参数含义:
        %90 (list): 输出，append后的list。
        %_output_size.1 (list): 需要进行append的list。
        %v.1 (-): append的元素。
    """
    scope_name = mapper.normalize_scope_name(node)
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    layer_outputs = [inputs_name[0]]
    # 获取当前节点输出的list
    current_outputs = [inputs_name[0]]
    # 处理输入0，即_output_size.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["list"] = inputs_name[0]
    # 处理输入1，即v.1
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["element"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.append",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_arange(mapper, graph, node):
    """ 构造以步长均匀分隔给定数值区间的PaddleLayer。
    TorchScript示例:
        有三种情况，分别处理。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    current_inputs = []
    if len(inputs_name) == 5:
        # %position_ids.1 : Tensor = aten::arange(%52, %43, %45, %42, %46)
        # 输入的后三者分别代表layout、device、是否使用梯度
        # 处理输入0，即%52，代表end
        if inputs_name[0] in mapper.attrs:
            layer_attrs["end"] = mapper.attrs[inputs_name[0]]
        else:
            mapper._check_input(graph, inputs_node[0], inputs_name[0],
                                current_outputs, scope_name)
            layer_inputs["end"] = inputs_name[0]
            current_inputs.append(inputs_name[0])
        # 处理输入1，即%43，代表dtype
        if mapper.attrs[inputs_name[1]] is None:
            layer_attrs["dtype"] = None
        else:
            layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[1]]]
    elif len(inputs_name) == 6:
        # %position_ids.1 : Tensor = aten::arange(%51, %52, %43, %45, %42, %46)
        # 输入的后三者分别代表layout、device、是否使用梯度
        # 处理输入0，即%51，代表start
        if inputs_name[0] in mapper.attrs:
            layer_attrs["start"] = mapper.attrs[inputs_name[0]]
        else:
            mapper._check_input(graph, inputs_node[0], inputs_name[0],
                                current_outputs, scope_name)
            layer_inputs["start"] = inputs_name[0]
            current_inputs.append(inputs_name[0])
        # 处理输入1，即%52，代表end
        if inputs_name[1] in mapper.attrs:
            layer_attrs["end"] = mapper.attrs[inputs_name[1]]
        else:
            mapper._check_input(graph, inputs_node[1], inputs_name[1],
                                current_outputs, scope_name)
            layer_inputs["end"] = inputs_name[1]
            current_inputs.append(inputs_name[1])
        # 处理输入2，即%43，代表dtype
        if mapper.attrs[inputs_name[2]] is None:
            layer_attrs["dtype"] = None
        else:
            layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[2]]]
    elif len(inputs_name) == 7:
        # %position_ids.1 : Tensor = aten::arange(%51, %52, %53, %43, %45, %42, %46)
        # 输入的后三者分别代表layout、device、是否使用梯度
        # 处理输入0，即%51，代表start
        if inputs_name[0] in mapper.attrs:
            layer_attrs["start"] = mapper.attrs[inputs_name[0]]
        else:
            mapper._check_input(graph, inputs_node[0], inputs_name[0],
                                current_outputs, scope_name)
            layer_inputs["start"] = inputs_name[0]
            current_inputs.append(inputs_name[0])
        # 处理输入1，即%52，代表end
        if inputs_name[1] in mapper.attrs:
            layer_attrs["end"] = mapper.attrs[inputs_name[1]]
        else:
            mapper._check_input(graph, inputs_node[1], inputs_name[1],
                                current_outputs, scope_name)
            layer_inputs["end"] = inputs_name[1]
            current_inputs.append(inputs_name[1])
        # 处理输入2，即%53，代表step
        if inputs_name[2] in mapper.attrs:
            layer_attrs["step"] = mapper.attrs[inputs_name[2]]
        else:
            mapper._check_input(graph, inputs_node[2], inputs_name[2],
                                current_outputs, scope_name)
            layer_inputs["step"] = inputs_name[2]
            current_inputs.append(inputs_name[2])
        # 处理输入3，即%43，代表dtype
        if mapper.attrs[inputs_name[3]] is None:
            layer_attrs["dtype"] = None
        else:
            layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[3]]]
    else:
        raise Exception("Unknown aten::arange signature taking " + str(
            len(inputs_name)) + " arguments.")

    graph.add_layer(
        "paddle.arange",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_avg_pool2d(mapper, graph, node):
    """ 构造最大池化的PaddleLayer。
    TorchScript示例:
        %branch_pool.2 : Tensor = aten::avg_pool2d(%x.43, %538, %539, %540, %273, %272, %271)
        参数含义:
        %branch_pool.2 (Tensor): 输出，池化后的结果。
        %x.43 (Tensor): 需要池化的Tensor。
        %538 (list): 池化kernel的大小。
        %539 (list): 步长大小。
        %540 (list): 填充大小。
        %273 (bool): 是否用ceil函数计算输出高度和宽度。
        %272 (bool): 是否在平均池化模式不忽略填充值，False为忽略。
        %271 (int): 如果指定，它将用作除数，否则将使用池化区域的大小。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("pool2d", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.34
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%538
    layer_attrs["kernel_size"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%539
    layer_attrs["stride"] = mapper.attrs[inputs_name[2]]
    # 处理输入3，即%540
    layer_attrs["padding"] = mapper.attrs[inputs_name[3]]
    # 处理输入4，即%273
    layer_attrs["ceil_mode"] = mapper.attrs[inputs_name[4]]
    # 处理输入5，即%272
    layer_attrs["exclusive"] = not mapper.attrs[inputs_name[5]]
    # 处理输入6，即%271
    graph.add_layer(
        "prim.assert",
        inputs={},
        outputs=[inputs_name[6] + "_assert"],
        scope_name=scope_name if scope_name == "" else scope_name + "_assert",
        type="eq",
        key=mapper.attrs[inputs_name[6]],
        value=None)

    graph.add_layer(
        kernel="paddle.nn.AvgPool2D",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)

    return current_inputs, current_outputs


def aten_avg_pool3d(mapper, graph, node):
    """ 构造最大池化的PaddleLayer。
    TorchScript示例:
        %branch_pool.2 : Tensor = aten::avg_pool2d(%x.43, %538, %539, %540, %273, %272, %271)
        参数含义:
        %branch_pool.2 (Tensor): 输出，池化后的结果。
        %x.43 (Tensor): 需要池化的Tensor。
        %538 (list): 池化kernel的大小。
        %539 (list): 步长大小。
        %540 (list): 填充大小。
        %273 (bool): 是否用ceil函数计算输出高度和宽度。
        %272 (bool): 是否在平均池化模式不忽略填充值，False为忽略。
        %271 (int): 如果指定，它将用作除数，否则将使用池化区域的大小。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("pool2d", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.34
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%538
    layer_attrs["kernel_size"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%539
    layer_attrs["stride"] = mapper.attrs[inputs_name[2]]
    # 处理输入3，即%540
    layer_attrs["padding"] = mapper.attrs[inputs_name[3]]
    # 处理输入4，即%273
    layer_attrs["ceil_mode"] = mapper.attrs[inputs_name[4]]
    # 处理输入5，即%272
    layer_attrs["exclusive"] = not mapper.attrs[inputs_name[5]]
    # 处理输入6，即%271
    graph.add_layer(
        "prim.assert",
        inputs={},
        outputs=[inputs_name[6] + "_assert"],
        scope_name=scope_name if scope_name == "" else scope_name + "_assert",
        type="eq",
        key=mapper.attrs[inputs_name[6]],
        value=None)

    graph.add_layer(
        kernel="paddle.nn.AvgPool3D",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_avg_pool1d(mapper, graph, node):
    """ 构造最大池化的PaddleLayer。
    TorchScript示例:
        %branch_pool.2 : Tensor = aten::avg_pool1d(%x.43, %538, %539, %540, %273, %272, %271)
        参数含义:
        %branch_pool.2 (Tensor): 输出，池化后的结果。
        %x.43 (Tensor): 需要池化的Tensor。
        %538 (list): 池化kernel的大小。
        %539 (list): 步长大小。
        %540 (list): 填充大小。
        %273 (bool): 是否用ceil函数计算输出高度和宽度。
        %272 (bool): 是否在平均池化模式不忽略填充值，False为忽略。
        %271 (int): 如果指定，它将用作除数，否则将使用池化区域的大小。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("pool2d", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.34
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%538
    layer_attrs["kernel_size"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%539
    layer_attrs["stride"] = mapper.attrs[inputs_name[2]]
    # 处理输入3，即%540
    layer_attrs["padding"] = mapper.attrs[inputs_name[3]]
    # 处理输入4，即%273
    layer_attrs["ceil_mode"] = mapper.attrs[inputs_name[4]]
    # 处理输入5，即%272
    layer_attrs["exclusive"] = not mapper.attrs[inputs_name[5]]
    # 处理输入6，即%271
    graph.add_layer(
        "prim.assert",
        inputs={},
        outputs=[inputs_name[6] + "_assert"],
        scope_name=scope_name if scope_name == "" else scope_name + "_assert",
        type="eq",
        key=mapper.attrs[inputs_name[6]],
        value=None)

    graph.add_layer(
        kernel="paddle.nn.AvgPool1D",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_batch_norm(mapper, graph, node):
    """ 构造BatchNorm的PaddleLayer。
    TorchScript示例:
        %input.81 : Tensor = aten::batch_norm(%input.80, %778, %779, %776, %777, %780,
                                              %exponential_average_factor.23, %766, %781)
        参数含义:
        %input.81 (Tensor): 输出，批处理后的结果。
        %input.80 (Tensor): 需要进行批处理的特征层。
        %778 (Tensor): weights。
        %779 (Tensor): bias。
        %776 (Tensor): 全局均值。
        %777 (Tensor): 全局方差。
        %780 (bool): 是否训练。
        %exponential_average_factor.23 (float): 用于计算均值和方差的比例。
        %766 (float): 为了数值稳定加在分母上的值。
        %781 (bool): 是否启用cudnn。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("batchnorm", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    layer_attrs["is_test"] = True
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.80
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%778
    weights = mapper.pytorch_params[inputs_name[1]]
    mapper.paddle_params[op_name + ".weight"] = weights
    layer_attrs['num_channels'] = weights.shape[0]
    # 处理输入2，即%779
    if inputs_name[2] in mapper.pytorch_params:
        bias = mapper.pytorch_params[inputs_name[2]]
        if bias is not None:
            mapper.paddle_params[op_name + ".bias"] = bias
    else:
        mapper.paddle_params[op_name + ".bias"] = False
    # 处理输入3，即%776
    mean = mapper.pytorch_params[inputs_name[3]]
    mapper.paddle_params[op_name + "._mean"] = mean
    # 处理输入4，即%777
    var = mapper.pytorch_params[inputs_name[4]]
    mapper.paddle_params[op_name + "._variance"] = var
    # 处理输入6，即%exponential_average_factor.23
    layer_attrs["momentum"] = mapper.attrs[inputs_name[6]]
    # 处理输入7，即%766
    layer_attrs["epsilon"] = mapper.attrs[inputs_name[7]]

    graph.add_layer(
        "paddle.nn.BatchNorm",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_bmm(mapper, graph, node):
    """ 构造矩阵相乘的PaddleLayer。
    TorchScript示例:
        %x.222 : Tensor = aten::bmm(%32, %7)
        参数含义:
        %x.222 (Tensor): 输出，矩阵相乘后的结果。
        %i.12 (list): 输入1。
        %7 (int): 输入2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%i.12
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%288
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.bmm",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_cat(mapper, graph, node):
    """ 构造连接Tensor的PaddleLayer。
    TorchScript示例:
        %x.222 : Tensor = aten::cat(%32, %7)
        参数含义:
        %x.222 (Tensor): 输出，连接后的结果。
        %i.12 (list): 需要连接的Tensor组成的list。
        %7 (int): 连接的轴。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%13
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%12
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["axis"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    graph.add_layer(
        "paddle.concat",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_chunk(mapper, graph, node):
    """构造分割Tensor的PaddleLayer。
    TorchScript示例:
        %724 : Tensor[] = aten::chunk(%input.170, %720, %719)
        参数含义:
        %724 (Tensor): 输出，分割后的结果。
        %input.170 (Tensor): 需要进行分割的Tensor。
        %720 (int): 分割的块数。
        %719 (int): 分割的维度。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.170
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%720
    if inputs_name[1] in mapper.attrs:
        layer_attrs["num_or_sections"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["num_or_sections"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入2，即%719
    if inputs_name[2] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[2]]
    else:
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs, scope_name)
        layer_inputs["axis"] = inputs_name[2]
        current_inputs.append(inputs_name[2])
    graph.add_layer(
        "paddle.split",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_clamp(mapper, graph, node):
    """ 构造元素剪裁的PaddleLayer。
    TorchScript示例:
        %56 : Tensor = aten::clamp(%input.1, %46, %48, %49)
        参数含义:
        %56 (Tensor): 输出，累加后的结果。
        %input.1 (Tensor): 输入，需要剪裁的Tensor。
        %46 (float/Tensor): 最小值。
        %48 (float/Tensor): 最大值。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%46
    if inputs_name[1] in mapper.attrs:
        layer_attrs["min"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["min"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入2，即%48，代表dtype
    if inputs_name[2] in mapper.attrs:
        layer_attrs["max"] = mapper.attrs[inputs_name[2]]
    else:
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs, scope_name)
        layer_inputs["max"] = inputs_name[2]
        current_inputs.append(inputs_name[2])

    graph.add_layer(
        "paddle.clip",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_clamp_min(mapper, graph, node):
    """ 构造元素剪裁的PaddleLayer。
    TorchScript示例:
        %56 : Tensor = aten::clamp_min(%input.1, %46)
        参数含义:
        %56 (Tensor): 输出，累加后的结果。
        %input.1 (Tensor): 输入，需要剪裁的Tensor。
        %46 (float/Tensor): 最小值。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%46
    if inputs_name[1] in mapper.attrs:
        layer_attrs["min"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["min"] = inputs_name[1]
        current_inputs.append(inputs_name[1])

    graph.add_layer(
        "paddle.clip",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten___contains__(mapper, graph, node):
    """ 构造in的PaddleLayer。
    TorchScript示例:
        %51 : bool = aten::__contains__(%50, %name.1)
        参数含义:
        %51 (bool): 输出，第一个元素是否包含第二个元素。
        %50 (-): 需对比的输入1。
        %name.1 (-): 需对比的输入2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%50
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 处理输入1，即%name.1
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["element"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.contain",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_constant_pad_nd(mapper, graph, node):
    """ 构造填充固定值的PaddleLayer。
    TorchScript示例:
        %58 : Tensor = aten::constant_pad_nd(%input1.24, %4876, %42)
        参数含义:
        %58 (Tensor): 输出，填充后的Tensor。
        %input1.24 (Tensor): 需要填充的Tensor。
        %4876 (list): 填充大小。
        %42 (-): 填充值。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("pad", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input1.24
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 处理输入1，即%4876
    is_padding_tensor = False
    if inputs_name[1] in mapper.attrs:
        layer_attrs["padding"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["pad"] = inputs_name[1]
        is_padding_tensor = True
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入2，即%42
    layer_attrs["value"] = mapper.attrs[inputs_name[2]]

    if not is_padding_tensor:
        graph.add_layer(
            "prim.shape",
            inputs={"input": inputs_name[0]},
            outputs=[inputs_name[0] + "_shape"],
            scope_name=scope_name)
        graph.add_layer(
            "prim.len",
            inputs={"input": inputs_name[0] + "_shape"},
            outputs=[inputs_name[0] + "_len"],
            scope_name=scope_name)

    def add_pad_layers(kernel, dim):
        graph.add_layer(
            "prim.ne",
            inputs={"x": inputs_name[0] + "_len"},
            outputs=[inputs_name[0] + "_cond"],
            scope_name=scope_name,
            y=dim)
        graph.add_layer(
            "prim.if", {'input': inputs_name[0] + "_cond"},
            outputs=[inputs_name[0] + "_if", output_name],
            scope_name=scope_name)
        if_layer = graph.layers[list(graph.layers.keys())[-1]]
        block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
        block.add_layer(
            "prim.sub",
            inputs={"y": inputs_name[0] + "_len"},
            outputs=[inputs_name[0] + "_len0"],
            scope_name=scope_name,
            alpha=1.0,
            x=dim)
        block.add_layer(
            "prim.len2list",
            inputs={"len": inputs_name[0] + "_len0"},
            outputs=[inputs_name[0] + "_list"],
            scope_name=scope_name)
        block.add_layer(
            "paddle.unsqueeze",
            inputs={"x": inputs_name[0],
                    "axis": inputs_name[0] + "_list"},
            outputs=[inputs_name[0] + "_var"],
            scope_name=scope_name)
        block.add_layer(
            kernel,
            inputs={"input": inputs_name[0] + "_var"},
            outputs=copy.deepcopy(layer_outputs),
            scope_name=scope_name,
            **layer_attrs)
        block.add_layer(
            "paddle.squeeze",
            inputs={"x": output_name,
                    "axis": inputs_name[0] + "_list"},
            outputs=[output_name],
            scope_name=scope_name)
        if_layer.add_block(block)
        block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
        layer_inputs["input"] = inputs_name[0]
        block.add_layer(
            kernel,
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name,
            **layer_attrs)
        if_layer.add_block(block)
        if_layer.inputs["input-0"] = inputs_name[0]
        if_layer.inputs["input-1"] = inputs_name[0] + "_len"

    if not is_padding_tensor:
        if len(layer_attrs["padding"]) == 2:
            layer_outputs[0] = layer_outputs[0].replace("pad", "pad1d")
            add_pad_layers("paddle.nn.Pad1D", 3)
        elif len(layer_attrs["padding"]) == 4:
            layer_outputs[0] = layer_outputs[0].replace("pad", "pad2d")
            add_pad_layers("paddle.nn.Pad2D", 4)
        elif len(layer_attrs["padding"]) == 6:
            layer_outputs[0] = layer_outputs[0].replace("pad", "pad3d")
            add_pad_layers("paddle.nn.Pad3D", 5)
        else:
            raise Exception("The lenght of padding list must be 2, 4 or 6!")
    else:
        graph.add_layer(
            "custom_layer:Pad",
            inputs=layer_inputs,
            outputs=[output_name],
            scope_name=scope_name,
            **layer_attrs)
    return current_inputs, current_outputs


def aten_contiguous(mapper, graph, node):
    """ 构造在内存中连续存储的PaddleLayer。
    TorchScript示例:
        %x.7 : Tensor = aten::contiguous(%4058, %4046)
        参数含义:
        %x.7 (Tensor): 输出，在内存中连续存储的Tensor。
        %4058 (Tensor): 原始Tensor。
        %4046 (int): 存储的形式。
    【注意】Paddle中无此用法，所以此处翻译成赋值。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%4058
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.equal",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_conv2d(mapper, graph, node):
    """ 构造conv2d的PaddleLayer。
    TorchScript示例:
        %input.10 : Tensor = aten::conv2d(%input.8, %25, %27, %28, %29, %30, %26)
        参数含义:
        %input.10 (Tensor): 输出，卷积后的结果。
        %input.8 (Tensor): 需要进行卷积的特征层。
        %25 (Tensor): weights。
        %27 (Tensor): bias。
        %28 (int): 步长大小。
        %29 (int): 填充大小。
        %30 (int): 空洞大小。
        %26 (int): 卷积的组数。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("conv2d", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.8
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%25
    weights = mapper.pytorch_params[inputs_name[1]]
    mapper.paddle_params[op_name + ".weight"] = weights
    layer_attrs["out_channels"] = weights.shape[0]
    layer_attrs["kernel_size"] = weights.shape[2:]
    # 处理输入2，即%27
    if inputs_name[2] in mapper.pytorch_params:
        bias = mapper.pytorch_params[inputs_name[2]]
        if bias is not None:
            mapper.paddle_params[op_name + ".bias"] = bias
        else:
            layer_attrs["bias_attr"] = False
    else:
        layer_attrs["bias_attr"] = False
    # 处理输入3，即%28
    layer_attrs["stride"] = mapper.attrs[inputs_name[3]]
    # 处理输入4，即%29
    layer_attrs["padding"] = mapper.attrs[inputs_name[4]]
    # 处理输入5，即%30
    layer_attrs["dilation"] = mapper.attrs[inputs_name[5]]
    # 处理输入6，即%26
    layer_attrs["groups"] = mapper.attrs[inputs_name[6]]
    layer_attrs['in_channels'] = weights.shape[1] * mapper.attrs[inputs_name[6]]

    graph.add_layer(
        "paddle.nn.Conv2D",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten__convolution(mapper, graph, node):
    """ 构造conv2d的PaddleLayer。
    TorchScript示例:
        %input.10 : Tensor = aten::_convolution(%input.1, %18, %10, %19, %20, %21, %13, %22, %12, %13, %13, %15)
        参数含义:
        %input.10 (Tensor): 输出，卷积后的结果。
        %input.8 (Tensor): 需要进行卷积的特征层。
        %18 (Tensor): weights。
        %10 (Tensor): bias。
        %19 (list): 步长大小。
        %20 (list): 填充大小。
        %21 (list): 空洞大小。
        %13 (bool): 是否进行转置卷积。
        %22 (list): 输出形状上一侧额外添加的大小。
        %12 (int): 卷积的组数。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("conv2d", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.8
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%18
    weights = mapper.pytorch_params[inputs_name[1]]
    mapper.paddle_params[op_name +
                         ".weight"] = weights  #np.swapaxes(weights, 0, 1)
    if mapper.attrs[inputs_name[6]]:
        layer_attrs["out_channels"] = weights.shape[1]
    else:
        layer_attrs["out_channels"] = weights.shape[0]
    layer_attrs["kernel_size"] = weights.shape[2:]
    # 处理输入2，即%10
    if inputs_name[2] in mapper.pytorch_params:
        bias = mapper.pytorch_params[inputs_name[2]]
        if bias is not None:
            mapper.paddle_params[op_name + ".bias"] = bias
        else:
            layer_attrs["bias_attr"] = False
    else:
        layer_attrs["bias_attr"] = False
    # 处理输入3，即%19
    layer_attrs["stride"] = mapper.attrs[inputs_name[3]]
    # 处理输入4，即%20
    layer_attrs["padding"] = mapper.attrs[inputs_name[4]]
    # 处理输入5，即%21
    layer_attrs["dilation"] = mapper.attrs[inputs_name[5]]
    # 处理输入6，即%13
    if mapper.attrs[inputs_name[6]]:
        # 处理输入7，即%22
        layer_attrs["output_padding"] = mapper.attrs[inputs_name[7]]
    # 处理输入8，即%12
    layer_attrs["groups"] = mapper.attrs[inputs_name[8]]
    if mapper.attrs[inputs_name[6]]:
        layer_attrs['in_channels'] = weights.shape[0] * mapper.attrs[
            inputs_name[8]]
    else:
        layer_attrs['in_channels'] = weights.shape[1] * mapper.attrs[
            inputs_name[8]]
    if mapper.attrs[inputs_name[6]]:
        graph.add_layer(
            "paddle.nn.Conv2DTranspose",
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name,
            **layer_attrs)
    else:
        graph.add_layer(
            "paddle.nn.Conv2D",
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name,
            **layer_attrs)
    return current_inputs, current_outputs


def aten_conv_transpose2d(mapper, graph, node):
    """ 构造conv_transpose2d的PaddleLayer。
    TorchScript示例:
        %input.10 : Tensor = aten::conv_transpose2d(%input.1, %18, %10, %19, %20, %21, %13, %22)
        参数含义:
        %input.10 (Tensor): 输出，卷积后的结果。
        %input.8 (Tensor): 需要进行卷积的特征层。
        %18 (Tensor): weights。
        %10 (Tensor): bias。
        %19 (list): 步长大小。
        %20 (list): 填充大小。
        %21 (int/tuple): 输出形状上一侧额外添加的大小。
        %13 (int): 二维卷积层的组数。
        %22 (int/tuple): 空洞大小。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("conv2d", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.8
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%18
    weights = mapper.pytorch_params[inputs_name[1]]
    mapper.paddle_params[op_name + ".weight"] = weights
    layer_attrs["out_channels"] = weights.shape[1]
    layer_attrs["kernel_size"] = weights.shape[2:]
    # 处理输入2，即%10
    if inputs_name[2] in mapper.pytorch_params:
        bias = mapper.pytorch_params[inputs_name[2]]
        if bias is not None:
            mapper.paddle_params[op_name + ".bias"] = bias
        else:
            layer_attrs["bias_attr"] = False
    else:
        layer_attrs["bias_attr"] = False
    # 处理输入3，即%19
    layer_attrs["stride"] = mapper.attrs[inputs_name[3]]
    # 处理输入4，即%20
    layer_attrs["padding"] = mapper.attrs[inputs_name[4]]
    # 处理输入5，即%21
    layer_attrs["output_padding"] = mapper.attrs[inputs_name[5]]
    # 处理输入6，即%13
    layer_attrs["groups"] = mapper.attrs[inputs_name[6]]
    # 处理输入7，即%22
    layer_attrs["dilation"] = mapper.attrs[inputs_name[7]]
    layer_attrs['in_channels'] = weights.shape[0] * mapper.attrs[inputs_name[6]]
    graph.add_layer(
        "paddle.nn.Conv2DTranspose",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_cos(mapper, graph, node):
    """ 构造数学计算cos的PaddleLayer。
    TorchScript示例:
        %94 : Tensor = aten::cos(%sinusoid_inp.1)
        参数含义:
        %94 (Tensor): 输出，cos之后的结果。
        %sinusoid_inp.1 (Tensor): 需要进行shape的Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%sinusoid_inp.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.cos",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_cumsum(mapper, graph, node):
    """ 构造与前一个元素累加的PaddleLayer。
    TorchScript示例:
        %56 : Tensor = aten::cumsum(%mask.1, %46, %48)
        参数含义:
        %56 (Tensor): 输出，累加后的结果。
        %mask.1 (Tensor): 输入，需要累加的Tensor。
        %46 (int): 累加的维度。
        %48 (int/None): Tensor的类型。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%mask.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%46
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["axis"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入1，即%48，代表dtype
    if mapper.attrs[inputs_name[2]] is None:
        layer_attrs["dtype"] = None
    else:
        layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[2]]]

    graph.add_layer(
        "paddle.cumsum",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_detach(mapper, graph, node):
    """ 构造返回一个新的Tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置的PaddleLayer。
    TorchScript示例:
        %107 : Tensor = aten::detach(%new_mem.1)
        参数含义:
        %107 (Tensor): 输出，得到的Scalar。
        %new_mem.1 (Tensor): 输入。
    【注意】由于Paddle无此操作，所以此处制转换为赋值。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%end.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    graph.add_layer(
        "prim.equal",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)

    return current_inputs, current_outputs


def aten_dict(mapper, graph, node):
    """ 构造初始化dict的PaddleLayer。
    TorchScript示例:
        %features.1 : Dict(str, Tensor) = aten::dict()
        参数含义:
        %features.1: 输出，初始化的dict。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    current_inputs = {}
    # 获取当前节点输出的list
    current_outputs = [output_name]

    graph.add_layer(
        "prim.dict",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_dim(mapper, graph, node):
    """ 构造获取维度的PaddleLayer。
    TorchScript示例:
        %106 : int = aten::dim(%101)
        参数含义:
        %106 (int): 输出，Tensor的维度。
        %101 (Tensor): 输入的Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.8
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.shape",
        inputs=layer_inputs,
        outputs=[output_name],
        scope_name=scope_name)
    graph.add_layer(
        "prim.len",
        inputs={"input": output_name},
        outputs=[output_name],
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_div_(mapper, graph, node):
    """ 构造除法的PaddleLayer。
    TorchScript示例:
        %bx_bw0.3 : Tensor = aten::div_(%bx_bw.3, %2678)
        参数含义:
        %bx_bw0.3 (-): 除后的结果。
        %bx_bw.3 (-): 被除数。
        %2678 (int): 除数。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%124
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%123
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.div",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_div(mapper, graph, node):
    """ 构造除法的PaddleLayer。
    TorchScript示例:
        %bx_bw0.3 : Tensor = aten::div_(%bx_bw.3, %2678)
        参数含义:
        %bx_bw0.3 (-): 除后的结果。
        %bx_bw.3 (-): 被除数。
        %2678 (int): 除数。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%124
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%123
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.div",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_dropout(mapper, graph, node):
    """ 构造Dropout的PaddleLayer。
    TorchScript示例:
        %119 : Tensor = aten::dropout(%result.3, %117, %118)
        参数含义:
        %119 (Tensor): Dropout后的Tensor。
        %result.3 (Tensor): 输入Tensor。
        %118 (bool): 是否是训练阶段。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("dropout", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%119
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.nn.Dropout",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        p=0.0)
    return current_inputs, current_outputs


def aten_dropout_(mapper, graph, node):
    """ 构造Dropout的PaddleLayer。
    TorchScript示例:
        %119 : Tensor = aten::dropout_(%result.3, %117, %118)
        参数含义:
        %119 (Tensor): Dropout后的Tensor。
        %result.3 (Tensor): 输入Tensor。
        %118 (bool): 是否是训练阶段。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("dropout", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%119
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.nn.Dropout",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        p=0.0)
    return current_inputs, current_outputs


def aten_embedding(mapper, graph, node):
    """ 构造embedding的PaddleLayer。
    TorchScript示例:
        %inputs_embeds.1 : Tensor = aten::embedding(%57, %input_ids.1, %45, %46, %46)
        参数含义:
        %inputs_embeds.1 (Tensor): 输出，embedding后的结果。
        %57 (Tensor): weights。
        %input_ids.1 (Tensor): 需要进行embedding的特征层。
        %45 (int): padding_idx。
        %46 (bool): scale_grad_by_freq。
        %46 (bool): sparse。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("embedding", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%57
    weights = mapper.pytorch_params[inputs_name[0]]
    mapper.paddle_params[op_name + ".weight"] = weights
    layer_attrs["num_embeddings"] = weights.shape[0]
    layer_attrs["embedding_dim"] = weights.shape[1]
    # 处理输入1，即%input_ids.1
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入2，即%45
    if mapper.attrs[inputs_name[2]] == -1:
        layer_attrs["padding_idx"] = None
    else:
        layer_attrs["padding_idx"] = mapper.attrs[inputs_name[2]]
    # 处理输入4，即%46
    layer_attrs["sparse"] = mapper.attrs[inputs_name[4]]

    graph.add_layer(
        "paddle.nn.Embedding",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_eq(mapper, graph, node):
    """ 构造判断数值是否相等的PaddleLayer。
    TorchScript示例:
        %125 : bool = aten::eq(%124, %123)
        参数含义:
        %125 (bool): 对比后结果。
        %124 (-): 需对比的输入1。
        %123 (-): 需对比的输入2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%124
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    x_value = list(node.inputs())[0]
    x_type = x_value.type()
    # 处理输入1，即%123
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    y_value = list(node.inputs())[1]
    y_type = y_value.type()
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    graph.add_layer(
        "prim.eq",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_erf(mapper, graph, node):
    """ 构造逐元素计算 Erf 激活函数的PaddleLayer。
    TorchScript示例:
        %94 : Tensor = aten::erf(%sinusoid_inp.1)
        参数含义:
        %94 (Tensor): 输出，erf之后的结果。
        %sinusoid_inp.1 (Tensor): 需要进行erf的Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%sinusoid_inp.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.erf",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_exp(mapper, graph, node):
    """ 构造以自然数e为底指数运算的PaddleLayer。
    TorchScript示例:
        %55 : Tensor = aten::tanh(%54)
        参数含义:
        %55 (Tensor): 输出，运算后的结果。
        %54 (Tensor): 需要指数运算的Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.exp",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_expand(mapper, graph, node):
    """ 构造对某维度进行广播的PaddleLayer。
    TorchScript示例:
        %1889 : Tensor = aten::expand(%1875, %1888, %1567)
        参数含义:
        %1889 (Tensor): 广播后的结果。
        %1875 (Tensor): 需要广播的Tensor。
        %1888 (int): 广播的维度。
        %1567 (bool): 未使用。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%1875
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%51
    if inputs_name[1] in mapper.attrs:
        layer_attrs["shape"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["shape"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    graph.add_layer(
        "paddle.expand",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_expand_as(mapper, graph, node):
    """ 构造广播的PaddleLayer。
    TorchScript示例:
        %1889 : Tensor = aten::expand_as(%1875, %1888)
        参数含义:
        %1889 (Tensor): 广播后的结果。
        %1875 (Tensor): 需要广播的Tensor。
        %1888 (Tensor): 广播的示例。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%1875
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%1888
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.type",
        inputs={"input": inputs_name[0]},
        outputs=[inputs_name[0] + "_type"],
        scope_name=scope_name)
    graph.add_layer(
        "prim.eq",
        inputs={"x": inputs_name[0] + "_type"},
        outputs=[inputs_name[0] + "_cond"],
        scope_name=scope_name,
        y=paddle_dtypes.t_bool)
    graph.add_layer(
        "prim.if", {'input': inputs_name[0] + "_cond"},
        outputs=[inputs_name[0] + "_if1", inputs_name[0]],
        scope_name=scope_name)
    if_layer = graph.layers[list(graph.layers.keys())[-1]]
    block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
    block.add_layer(
        "prim.type",
        inputs={"input": inputs_name[1]},
        outputs=[inputs_name[1] + "_type"],
        scope_name=scope_name)
    block.add_layer(
        "paddle.cast",
        inputs={"x": inputs_name[0]},
        outputs=[inputs_name[0]],
        scope_name=scope_name,
        dtype=inputs_name[1] + "_type")
    if_layer.add_block(block)
    block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
    if_layer.add_block(block)
    if_layer.inputs["input-0"] = inputs_name[0]
    if_layer.inputs["input-1"] = inputs_name[1]
    graph.add_layer(
        "paddle.expand_as",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    graph.add_layer(
        "prim.if", {'input': inputs_name[0] + "_cond"},
        outputs=[inputs_name[0] + "_if2", output_name],
        scope_name=scope_name)
    if_layer = graph.layers[list(graph.layers.keys())[-1]]
    block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
    block.add_layer(
        "paddle.cast",
        inputs={"x": layer_outputs[0]},
        outputs=copy.deepcopy(layer_outputs),
        scope_name=scope_name,
        dtype=string("bool"))
    if_layer.add_block(block)
    block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
    if_layer.add_block(block)
    if_layer.inputs["input-0"] = layer_outputs[0]
    # TODO(syf): check expand_as
    #     # 处理输入0，即%1875
    #     mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs, scope_name)
    #     layer_inputs["x"] = inputs_name[0]
    #     # 处理输入1，即%1888
    #     mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs, scope_name)
    #     layer_inputs["y"] = inputs_name[1]
    #     # 获取当前节点输入的list
    #     current_inputs = list(layer_inputs.values())
    #     graph.add_layer(
    #         "paddle.expand_as", inputs=layer_inputs, outputs=layer_outputs, scope_name=scope_name)
    return current_inputs, current_outputs


def aten_eye(mapper, graph, node):
    """ 构造批次二维矩阵的PaddleLayer。
    TorchScript示例:
        %68 : Tensor = aten::eye(%49, %_50, %_51, %15, %9, %67, %7)
        参数含义:
        %68 (Tensor): 输出，构造的矩阵。
        %49 (int): 行数。
        %_50 (int): 列数，非必须。
        %_51 (Tensor): 非必须。
        %9 (int): layout。
        %67 (str): 设备。
        %7 (bool): 是否计算梯度。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%49
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["num_rows"] = inputs_name[0]
    if len(inputs_name) > 5:
        # 处理输入1，即%_50
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["num_columns"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理倒数第4个输入，即%15
    layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[-4]]]

    graph.add_layer(
        "paddle.eye",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_feature_dropout(mapper, graph, node):
    """ 构造Dropout的PaddleLayer。
    TorchScript示例:
        %119 : Tensor = aten::feature_dropout(%result.3, %117, %118)
        参数含义:
        %119 (Tensor): Dropout后的Tensor。
        %result.3 (Tensor): 输入Tensor。
        %118 (bool): 是否是训练阶段。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("dropout", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%119
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.nn.Dropout",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        p=0.0)
    return current_inputs, current_outputs


def aten_flatten(mapper, graph, node):
    """ 构造flatten的PaddleLayer。
    TorchScript示例:
        %x.8 : Tensor = aten::flatten(%x, %4, %2)
        参数含义:
        %x.8 (Tensor): flatten后结果。
        %x (Tensor): 输入Tensor。
        %4 (int): flatten的开始维度。
        %2 (int): flatten的结束维度。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    # 处理输入1，即%4
    layer_attrs["start_axis"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%20
    layer_attrs["stop_axis"] = mapper.attrs[inputs_name[2]]
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.flatten",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_Float(mapper, graph, node):
    """ 构造取浮点型的PaddleLayer。
    TorchScript示例:
        %3992 : float = aten::Float(%3991)
        参数含义:
        %3992 (int): 向上取整后的整数。
        %3991 (float): 需要取整的浮点数。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%3991
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.float",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_floor(mapper, graph, node):
    """ 构造向上取整的PaddleLayer。
    TorchScript示例:
        %3978 : int = aten::floor(%scale.18)
        参数含义:
        %3978 (int): 向上取整后的整数。
        %scale.18 (float): 需要取整的浮点数。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%scale.18
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    graph.add_layer(
        "prim.type", {'input': inputs_name[0]},
        outputs=[inputs_name[0] + "_type"],
        scope_name=scope_name)
    graph.add_layer(
        "prim.str", {'input': inputs_name[0] + "_type"},
        outputs=[inputs_name[0] + "_type"],
        scope_name=scope_name)
    graph.add_layer(
        "prim.eq",
        inputs={"x": inputs_name[0] + "_type"},
        outputs=[inputs_name[0] + "_cond"],
        scope_name=scope_name,
        y=paddle_dtypes.t_bool)
    graph.add_layer(
        "prim.if", {'input': inputs_name[0] + "_cond"},
        outputs=[inputs_name[0] + "_if"],
        scope_name=scope_name)
    if_layer = graph.layers[list(graph.layers.keys())[-1]]
    block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
    block.add_layer(
        "paddle.floor",
        inputs=copy.deepcopy(layer_inputs),
        outputs=copy.deepcopy(layer_outputs),
        scope_name=scope_name)
    if_layer.add_block(block)
    block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
    block.add_layer(
        "prim.floor",
        inputs=copy.deepcopy(layer_inputs),
        outputs=copy.deepcopy(layer_outputs),
        scope_name=scope_name)
    if_layer.add_block(block)
    if_layer.inputs["input-0"] = inputs_name[0]
    if_layer.outputs.append(output_name)
    return current_inputs, current_outputs


def aten_floordiv(mapper, graph, node):
    """ 构造向上取整除法的PaddleLayer。
    TorchScript示例:
        %channels_per_group.2 : int = aten::floordiv(%num_channels.2, %3690)
        参数含义:
        %channels_per_group.2 (-): 除后的结果。
        %num_channels.2 (-): 被除数。
        %2 (int): 除数。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%124
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%123
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.floordiv",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_floor_divide(mapper, graph, node):
    """ 构造向上取整除法的PaddleLayer。
    TorchScript示例:
        %channels_per_group.2 : int = aten::floor_divide(%num_channels.2, %3690)
        参数含义:
        %channels_per_group.2 (-): 除后的结果。
        %num_channels.2 (-): 被除数。
        %2 (int): 除数。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%124
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%123
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.floordiv",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_full_like(mapper, graph, node):
    """ 构造创建一个与输入具有相同的形状并且数据类型固定的Tensor的PaddleLayer。
    TorchScript示例:
        %159 : Tensor = aten::full_like(%val_if_large.3, %51, %50, %62, %53, %65, %66)
        参数含义:
        %159 (Tensor): 输出，全为固定值的Tensor。
        %val_if_large.3 (Tensor): 类似形状的Tensor。
        %51 (int/float/bool): 填充值。
        %50 (int): dtype。
        %62 (int): layout。
        %53 (int): device。
        %65 (bool): 是否计算梯度。
        %66 (int): 内存形式。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%val_if_large.3
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%51
    if inputs_name[1] in mapper.attrs:
        layer_attrs["fill_value"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["fill_value"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入2，即%50，代表dtype
    layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[2]]]

    graph.add_layer(
        "paddle.full_like",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_gather(mapper, graph, node):
    """ 构造gather激活的PaddleLayer。
    TorchScript示例:
        %result.3 : Tensor = aten::gather(%input.5, %18, %19, %20, %21)
        参数含义:
        %result.3 (Tensor): 输出，gather后的结果。
        %result.5 (Tensor): 需要gather的Tensor。
        %18 (int): 需要gather的维度。
        %19 (Tensor): 需要gather的索引。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("gather", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%18
    layer_attrs["dim"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%19
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    layer_inputs["index"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "custom_layer:Gather",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_gelu(mapper, graph, node):
    """ 构造GeLU激活的PaddleLayer。
    TorchScript示例:
        %result.3 : Tensor = aten::gelu(%input.5)
        参数含义:
        %result.3 (Tensor): 输出，GELU后的结果。
        %result.5 (Tensor): 需要GELU的Tensor。
    注意: inplace这个参数在paddle中未实现
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("gelu", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.nn.GELU",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten___getitem__(mapper, graph, node):
    """ 构造获取list中元素的PaddleLayer。
    TorchScript示例:
        %v.1 : int = aten::__getitem__(%72, %88)
        参数含义:
        %v.1 (-): 输出，list中的元素。
        %72 (list): 需要获取元素的list。
        %88 (int): 索引。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%72
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["list"] = inputs_name[0]
    # 处理输入1，即%88
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["index"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.getitem",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_gt(mapper, graph, node):
    """ 构造对比大小的PaddleLayer。
    TorchScript示例:
        %83 : bool = aten::gt(%82, %78)
        参数含义:
        %83 (bool): 输出，第一个元素是否大于第二个元素。
        %82 (-): 需对比的输入1。
        %78 (-): 需对比的输入2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%82
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%78
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.gt",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_gru(mapper, graph, node):
    """ 构造门控循环单元网络（GRU）的PaddleLayer。
    TorchScript示例:
        %21, %22 = aten::gru(%input, %hx, %20, %11, %10, %9, %11, %8, %11)
        参数含义:
        %21 (Tensor): 输出，由前向和后向cell的输出拼接得到。
        %22 (Tensor): 输出，最终状态。
        %input (Tensor): 网络输入。
        %hx (Tensor): 网络的初始状态。
        %20 (list): 所有权重组合成的list。
        %11 (bool): 是否使用bias。
        %10 (int): 网络层数。
        %9 (float): dropout概率。
        %11 (bool): 是否为训练阶段。
        %8 (bool): 是否使用双向LSTM。
        %11 (bool): 第一个维度是否为batch size。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("gru", mapper.nn_name2id)
    output_names = mapper._get_outputs_name(node)
    layer_outputs = [op_name]
    layer_outputs.extend(output_names)
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = output_names
    # 处理输入0，即%input.95
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input0"] = inputs_name[0]
    # 处理输入1，即%734
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["input1"] = inputs_name[1]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入2，即%734
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    graph.layers.pop(mapper.output2id[inputs_name[2]])
    param_inputs_name, _ = mapper._get_inputs_name(inputs_node[2])
    new_param_inputs_name = list()
    for i, param_name in enumerate(param_inputs_name):
        if i == 0:
            layer_attrs["hidden_size"] = int(
                mapper.paddle_params[param_name].shape[0] / 3)
            layer_attrs["input_size"] = int(mapper.paddle_params[param_name]
                                            .shape[1])
        if len(mapper.paddle_params[param_name].shape) > 1:
            part_name = param_name.split("_weight_")[-1]
            mapper.paddle_params["{}.weight_{}".format(
                op_name, part_name)] = mapper.paddle_params[param_name]
            new_param_inputs_name.append("{}.weight_{}".format(op_name,
                                                               part_name))
        else:
            part_name = param_name.split("_bias_")[-1]
            mapper.paddle_params["{}.bias_{}".format(
                op_name, part_name)] = mapper.paddle_params[param_name]
        mapper.paddle_params.pop(param_name)

    # 处理输入3，即%526
    is_bias = mapper.attrs[inputs_name[3]]
    if not is_bias:
        for param_name in new_param_inputs_name:
            bias_name = param_name.replace("weight", "bias")
            bias_shape = mapper.paddle_params[param_name].shape[:1]
            mapper.paddle_params[bias_name] = np.zeros(bias_shape).astype(
                "float32")
    # 处理输入4，即%525
    layer_attrs["num_layers"] = mapper.attrs[inputs_name[4]]
    # 处理输入5，即%524
    layer_attrs["dropout"] = mapper.attrs[inputs_name[5]]
    # 处理输入7，即%526
    is_bidirectional = mapper.attrs[inputs_name[7]]
    if is_bidirectional:
        layer_attrs["direction"] = string("bidirectional")
    # 处理输入8，即%526
    batch_first = mapper.attrs[inputs_name[8]]
    if not batch_first:
        layer_attrs["time_major"] = True
    graph.add_layer(
        "paddle.nn.GRU",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_hardtanh_(mapper, graph, node):
    """ 构造hardtanh激活的PaddleLayer。
    TorchScript示例:
        %result.9 : Tensor = aten::hardtanh_(%input.20, %67, %66)
        参数含义:
        %result.9 (Tensor): 输出，hardtanh激活后的Tensor。
        %input.20 (Tensor): 需要hardtanh激活的Tensor。
        %67 (float): hardtanh激活的最小阈值。
        %66 (float): hardtanh激活的最大阈值。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("hardtanh", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.20
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%67
    layer_attrs["min"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%66
    layer_attrs["max"] = mapper.attrs[inputs_name[2]]

    if layer_attrs["min"] == 0 and layer_attrs["max"] == 6:
        graph.add_layer(
            "paddle.nn.ReLU6",
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name)
    else:
        graph.add_layer(
            'paddle.nn.Hardtanh',
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name,
            **layer_attrs)
    return current_inputs, current_outputs


def aten_index(mapper, graph, node):
    """ 构造选择元素的PaddleLayer。
    TorchScript示例:
        %1681 : Float = aten::index(%1653, %1680)
        参数含义:
        %1681 (Tensor): 输出，选择后的Tensor。
        %1653 (Tensor): 需要选择的Tensor。
        %1680 (int): 选择的索引。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%1653
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%1680
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["index"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.getitem",
        inputs={"list": layer_inputs["index"]},
        outputs=[layer_inputs["index"]],
        scope_name=scope_name,
        index=0)
    graph.add_layer(
        "paddle.index_select",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_index_select(mapper, graph, node):
    """ 构造选择元素的PaddleLayer。
    TorchScript示例:
        %bd.3 : Tensor = aten::index_select(%x2.3, %320, %371)
        参数含义:
        %bd.3 (Tensor): 输出，选择后的Tensor。
        %x2.3 (Tensor): 需要选择的Tensor。
        %320 (int): 维度。
        %371 (Tensor): 选择的索引。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x2.3
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%320
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["axis"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入2，即%371
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    layer_inputs["index"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.index_select",
        inputs=layer_inputs,
        outputs=current_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_instance_norm(mapper, graph, node):
    """构造InstanceNorm的PaddleLayer
    TorchScript示例:
        %res.7 : Tensor = aten::instance_norm(%res.5, %88, %85, %84, %83, %87, %91, %92, %87)
        参数含义:
        %res.7 (Tensor): 输出，InstanceNorm的结果。
        %res.5 (Tensor): 需要进行InstanceNorm的特征层。
        %88 (Tensor): weights。
        %85 (Tensor): bias。
        %84 (Tensor): 全局均值。
        %83 (Tensor): 全局方差。
        %87 (bool): 是否使用输入的统计。
        %91 (float): momentum。
        %92 (float): eps。
        %87 (bool): 是否启用cudnn。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("instance_norm", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.80
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%88
    if inputs_name[1] in mapper.pytorch_params:
        weights = mapper.pytorch_params[inputs_name[1]]
        mapper.paddle_params[op_name + ".scale"] = weights
        layer_attrs['num_features'] = weights.shape[0]
    # 处理输入2，即%85
    if inputs_name[2] in mapper.pytorch_params:
        bias = mapper.pytorch_params[inputs_name[2]]
        mapper.paddle_params[op_name + ".bias"] = bias
    # 处理输入3，即%84
    if inputs_name[3] in mapper.pytorch_params:
        mean = mapper.pytorch_params[inputs_name[3]]
        mapper.paddle_params[op_name + "._mean"] = mean
    # 处理输入4，即%83
    if inputs_name[4] in mapper.pytorch_params:
        var = mapper.pytorch_params[inputs_name[4]]
        mapper.paddle_params[op_name + "._variance"] = var
    # 处理输入6，即%91
    layer_attrs["momentum"] = 1 - mapper.attrs[inputs_name[6]]
    # 处理输入7，即%92
    layer_attrs["epsilon"] = mapper.attrs[inputs_name[7]]

    graph.add_layer(
        "custom_layer:InstanceNorm",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_Int(mapper, graph, node):
    """ 构造强转为int的PaddleLayer。
    TorchScript示例:
        %1739 : int = aten::Int(%1738)
        参数含义:
        %1739 (int): 输出，int型数据。
        %1738 (-): 需要强转的数据。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%1738
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.int",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten___is__(mapper, graph, node):
    """ 构造is not的PaddleLayer。
    TorchScript示例:
        %3949 : bool = aten::__isnot__(%size.122, %3931)
        参数含义:
        %3949 (bool): 输出，第一个元素是否不是第二个元素。
        %size.122 (-): 需对比的输入1。
        %3931 (-): 需对比的输入2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%size.122
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%3931
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.is",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten___isnot__(mapper, graph, node):
    """ 构造is not的PaddleLayer。
    TorchScript示例:
        %3949 : bool = aten::__isnot__(%size.122, %3931)
        参数含义:
        %3949 (bool): 输出，第一个元素是否不是第二个元素。
        %size.122 (-): 需对比的输入1。
        %3931 (-): 需对比的输入2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%size.122
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%3931
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.isnot",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_layer_norm(mapper, graph, node):
    """ 构造层归一化的PaddleLayer。
    TorchScript示例:
        %input0.4 : Tensor = aten::layer_norm(%input.6, %1181, %174, %173, %70, %71)
        参数含义:
        %input0.4 (Tensor): 输出，层归一化后的结果。
        %input.6 (Tensor): 需要进行层归一化的特征层。
        %1181 (list/int/tuple): 需规范化的shape。
        %174 (Tensor): weights。
        %173 (Tensor): bias。
        %70 (float): 指明在计算过程中是否添加较小的值到方差中以防止除零。
        %71 (bool): 是否启用cudnn。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("layernorm", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.6
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%1181
    layer_attrs["normalized_shape"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%174
    weights = mapper.pytorch_params[inputs_name[2]]
    mapper.paddle_params[op_name + ".weight"] = weights
    # 处理输入3，即%173
    if inputs_name[3] in mapper.pytorch_params:
        bias = mapper.pytorch_params[inputs_name[3]]
        if bias is not None:
            mapper.paddle_params[op_name + ".bias"] = bias
    else:
        mapper.paddle_params[op_name + ".bias"] = False
    # 处理输入4，即%70
    layer_attrs["epsilon"] = mapper.attrs[inputs_name[4]]

    graph.add_layer(
        "paddle.nn.LayerNorm",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_le(mapper, graph, node):
    """ 构造对比大小的PaddleLayer。
    TorchScript示例:
        %80 : bool = aten::le(%78, %79)
        参数含义:
        %80 (bool): 输出，第一个元素是否小于等于第二个元素。
        %78 (-): 需对比的输入1。
        %79 (-): 需对比的输入2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%78
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%79
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.le",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_leaky_relu_(mapper, graph, node):
    """ 构造leaky relu激活的PaddleLayer。
    TorchScript示例:
        %input.117 : Tensor = aten::leaky_relu_(%input.114, %1570)
        参数含义:
        %input.117 (Tensor): 输出，leaky relu后的结果。
        %input.114 (Tensor): 需要leaky relu的Tensor。
        %1570 (float): 输入中的元素小于0时的斜率。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("leakly_relu", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%1570
    layer_attrs["negative_slope"] = mapper.attrs[inputs_name[1]]

    graph.add_layer(
        "paddle.nn.LeakyReLU",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_leaky_relu(mapper, graph, node):
    """ 构造leaky relu激活的PaddleLayer。
    TorchScript示例:
        %input.117 : Tensor = aten::leaky_relu(%input.114, %1570)
        参数含义:
        %input.117 (Tensor): 输出，leaky relu后的结果。
        %input.114 (Tensor): 需要leaky relu的Tensor。
        %1570 (float): 输入中的元素小于0时的斜率。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("leakly_relu", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%1570
    layer_attrs["negative_slope"] = mapper.attrs[inputs_name[1]]

    graph.add_layer(
        "paddle.nn.LeakyReLU",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_len(mapper, graph, node):
    """ 构造获取list长度的PaddleLayer。
    TorchScript示例:
        %85 : int = aten::len(%83)
        参数含义:
        %85 (int): 输出，list的长度。
        %72 (list): 需要获取长度的list。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%72
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.len",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_log(mapper, graph, node):
    """ 构构造log的PaddleLayer。
    TorchScript示例:
        %787 : Tensor = aten::log(%786)
        参数含义:
        %787 (Tensor): 输出，取log的Tensor。
        %786 (Tensor): 需要获取log的Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%786
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.log",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_log_softmax(mapper, graph, node):
    """ 构造log_softmax的PaddleLayer。
    TorchScript示例:
        %4 = aten::log_softmax(%input, %2, %3)
        参数含义:
        %4 (Tensor): 输出的Tensor。
        %input (Tensor): 输入的Tensor。
        %2 (int): 指定对输入进行运算的轴。
        %3 (int): 输入Tensor的数据类型。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    current_inputs = []
    # 处理输入0，即%input
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%2，代表dtype
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["axis"] = inputs_name[1]
    # 处理输入2，即%3，代表dtype
    if mapper.attrs[inputs_name[2]] is not None:
        layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[2]]]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.nn.functional.log_softmax",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_lstm(mapper, graph, node):
    """ 构造长短期记忆网络（LSTM）的PaddleLayer。
    TorchScript示例:
        %input.96, %551, %552 = aten::lstm(%input.95, %734, %549, %526, %525, %524, %526, %526, %526)
        参数含义:
        %input.96 (Tensor): 输出，由前向和后向cell的输出拼接得到。
        %551 (Tensor): cell state。
        %552 (Tensor): hidden state。
        %input.95 (Tensor): 网络输入。
        %734 (Tensor): 网络的初始状态。
        %549 (list): 所有权重组合成的list。
        %526 (bool): 是否使用bias。
        %525 (int): 网络层数。
        %524 (float): dropout概率。
        %526 (bool): 是否为训练阶段。
        %526 (bool): 是否使用双向LSTM。
        %526 (bool): 第一个维度是否为batch size。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("lstm", mapper.nn_name2id)
    output_names = mapper._get_outputs_name(node)
    layer_outputs = [op_name]
    layer_outputs.extend(output_names)
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = output_names
    # 处理输入0，即%input.95
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input0"] = inputs_name[0]
    # 处理输入1，即%734
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["input1"] = inputs_name[1]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入2，即%734
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    graph.layers.pop(mapper.output2id[inputs_name[2]])
    param_inputs_name, _ = mapper._get_inputs_name(inputs_node[2])
    new_param_inputs_name = list()
    for i, param_name in enumerate(param_inputs_name):
        if i == 0:
            layer_attrs["hidden_size"] = int(
                mapper.paddle_params[param_name].shape[0] / 4)
            layer_attrs["input_size"] = int(mapper.paddle_params[param_name]
                                            .shape[1])
        if len(mapper.paddle_params[param_name].shape) > 1:
            part_name = param_name.split("_weight_")[-1]
            mapper.paddle_params["{}.weight_{}".format(
                op_name, part_name)] = mapper.paddle_params[param_name]
            new_param_inputs_name.append("{}.weight_{}".format(op_name,
                                                               part_name))
        else:
            part_name = param_name.split("_bias_")[-1]
            mapper.paddle_params["{}.bias_{}".format(
                op_name, part_name)] = mapper.paddle_params[param_name]
        mapper.paddle_params.pop(param_name)

    # 处理输入3，即%526
    is_bias = mapper.attrs[inputs_name[3]]
    if not is_bias:
        for param_name in new_param_inputs_name:
            bias_name = param_name.replace("weight", "bias")
            bias_shape = mapper.paddle_params[param_name].shape[:1]
            mapper.paddle_params[bias_name] = np.zeros(bias_shape).astype(
                "float32")
    # 处理输入4，即%525
    layer_attrs["num_layers"] = mapper.attrs[inputs_name[4]]
    # 处理输入5，即%524
    layer_attrs["dropout"] = mapper.attrs[inputs_name[5]]
    # 处理输入7，即%526
    is_bidirectional = mapper.attrs[inputs_name[7]]
    if is_bidirectional:
        layer_attrs["direction"] = string("bidirectional")
    # 处理输入8，即%526
    batch_first = mapper.attrs[inputs_name[8]]
    if not batch_first:
        layer_attrs["time_major"] = True
    graph.add_layer(
        "paddle.nn.LSTM",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_lt(mapper, graph, node):
    """ 构造对比大小的PaddleLayer。
    TorchScript示例:
        %80 : bool = aten::lt(%78, %79)
        参数含义:
        %80 (bool): 输出，第一个元素是否小于第二个元素。
        %78 (-): 需对比的输入1。
        %79 (-): 需对比的输入2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%78
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%79
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.lt",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_masked_fill_(mapper, graph, node):
    """ 构造填充mask的PaddleLayer。
    TorchScript示例:
        %input.4 : Tensor = aten::masked_fill_(%scores.2, %mask.2, %46)
        参数含义:
        %input.4 (Tensor): 输出，填充后的结果。
        %scores.2 (Tensor): 需要填充的Tensor。
        %mask.2 (Tensor): bool型的Tensor，哪些位置需要填充。
        %46 (-): 填充的值。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输入的list
    current_inputs = []
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.4
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    current_inputs.append(inputs_name[0])
    graph.add_layer(
        "prim.type",
        inputs={"input": inputs_name[0]},
        outputs=[inputs_name[0] + "_type"],
        scope_name=scope_name)
    # 处理输入1，即%scores.2
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    current_inputs.append(inputs_name[1])
    graph.add_layer(
        "paddle.logical_not",
        inputs={"x": inputs_name[1]},
        outputs=[inputs_name[1] + "_not"],
        scope_name=scope_name)
    graph.add_layer(
        "paddle.cast",
        inputs={"x": inputs_name[1]},
        outputs=[inputs_name[1] + "_mask"],
        scope_name=scope_name,
        dtype=inputs_name[0] + "_type")
    graph.add_layer(
        "paddle.cast",
        inputs={"x": inputs_name[1] + "_not"},
        outputs=[inputs_name[1] + "_not_mask"],
        scope_name=scope_name,
        dtype=inputs_name[0] + "_type")
    graph.add_layer(
        "paddle.multiply",
        inputs={"x": inputs_name[0],
                "y": inputs_name[1] + "_not_mask"},
        outputs=[inputs_name[0] + "_not_mask"],
        scope_name=scope_name)
    # 处理输入2，即%46
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    graph.add_layer(
        "prim.eq",
        inputs={"x": inputs_name[2]},
        outputs=[inputs_name[2] + "_cond1"],
        scope_name=scope_name,
        y="-float('inf')")
    graph.add_layer(
        "prim.eq",
        inputs={"x": inputs_name[2]},
        outputs=[inputs_name[2] + "_cond2"],
        scope_name=scope_name,
        y="float('inf')")
    graph.add_layer(
        "prim.or",
        inputs={
            "x": inputs_name[2] + "_cond1",
            "y": inputs_name[2] + "_cond2"
        },
        outputs=[inputs_name[2] + "_cond"],
        scope_name=scope_name)
    graph.add_layer(
        "prim.if", {'input': inputs_name[2] + "_cond"},
        outputs=[inputs_name[2] + "_if"],
        scope_name=scope_name)
    if_layer = graph.layers[list(graph.layers.keys())[-1]]
    block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
    block.add_layer(
        "prim.equal",
        inputs={"input": inputs_name[1] + "_mask"},
        outputs=[inputs_name[2] + "_1"],
        scope_name=scope_name)
    if_layer.add_block(block)
    block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
    block.add_layer(
        "prim.mul",
        inputs={"x": inputs_name[1] + "_mask",
                "y": inputs_name[2]},
        outputs=[inputs_name[2] + "_1"],
        scope_name=scope_name)
    if_layer.add_block(block)
    if_layer.inputs["input-0"] = inputs_name[1] + "_mask"
    if_layer.inputs["input-1"] = inputs_name[2]
    if_layer.outputs.append(inputs_name[2] + "_1")
    graph.add_layer(
        "paddle.add",
        inputs={"x": inputs_name[2] + "_1",
                "y": inputs_name[0] + "_not_mask"},
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_masked_fill(mapper, graph, node):
    """ 构造填充mask的PaddleLayer。
    TorchScript示例:
        %input.4 : Tensor = aten::masked_fill(%scores.2, %mask.2, %46)
        参数含义:
        %input.4 (Tensor): 输出，填充后的结果。
        %scores.2 (Tensor): 需要填充的Tensor。
        %mask.2 (Tensor): bool型的Tensor，哪些位置需要填充。
        %46 (-): 填充的值。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输入的list
    current_inputs = []
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.4
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    current_inputs.append(inputs_name[0])
    graph.add_layer(
        "prim.type",
        inputs={"input": inputs_name[0]},
        outputs=[inputs_name[0] + "_type"],
        scope_name=scope_name)
    # 处理输入1，即%scores.2
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    current_inputs.append(inputs_name[1])
    graph.add_layer(
        "paddle.logical_not",
        inputs={"x": inputs_name[1]},
        outputs=[inputs_name[1] + "_not"],
        scope_name=scope_name)
    graph.add_layer(
        "paddle.cast",
        inputs={"x": inputs_name[1]},
        outputs=[inputs_name[1] + "_mask"],
        scope_name=scope_name,
        dtype=inputs_name[0] + "_type")
    graph.add_layer(
        "paddle.cast",
        inputs={"x": inputs_name[1] + "_not"},
        outputs=[inputs_name[1] + "_not_mask"],
        scope_name=scope_name,
        dtype=inputs_name[0] + "_type")
    graph.add_layer(
        "paddle.multiply",
        inputs={"x": inputs_name[0],
                "y": inputs_name[1] + "_not_mask"},
        outputs=[inputs_name[0] + "_not_mask"],
        scope_name=scope_name)
    # 处理输入2，即%46
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    graph.add_layer(
        "prim.eq",
        inputs={"x": inputs_name[2]},
        outputs=[inputs_name[2] + "_cond1"],
        scope_name=scope_name,
        y="-float('inf')")
    graph.add_layer(
        "prim.eq",
        inputs={"x": inputs_name[2]},
        outputs=[inputs_name[2] + "_cond2"],
        scope_name=scope_name,
        y="float('inf')")
    graph.add_layer(
        "prim.or",
        inputs={
            "x": inputs_name[2] + "_cond1",
            "y": inputs_name[2] + "_cond2"
        },
        outputs=[inputs_name[2] + "_cond"],
        scope_name=scope_name)
    graph.add_layer(
        "prim.if", {'input': inputs_name[2] + "_cond"},
        outputs=[inputs_name[2] + "_if"],
        scope_name=scope_name)
    if_layer = graph.layers[list(graph.layers.keys())[-1]]
    block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
    block.add_layer(
        "prim.equal",
        inputs={"input": inputs_name[1] + "_mask"},
        outputs=[inputs_name[2] + "_1"],
        scope_name=scope_name)
    if_layer.add_block(block)
    block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
    block.add_layer(
        "prim.mul",
        inputs={"x": inputs_name[1] + "_mask",
                "y": inputs_name[2]},
        outputs=[inputs_name[2] + "_1"],
        scope_name=scope_name)
    if_layer.add_block(block)
    if_layer.inputs["input-0"] = inputs_name[1] + "_mask"
    if_layer.inputs["input-1"] = inputs_name[2]
    if_layer.outputs.append(inputs_name[2] + "_1")
    graph.add_layer(
        "paddle.add",
        inputs={"x": inputs_name[2] + "_1",
                "y": inputs_name[0] + "_not_mask"},
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_max(mapper, graph, node):
    """ 构造获取最大值的PaddleLayer。
    TorchScript示例:
        %val_if_large0.3 : Tensor = aten::max(%val_if_large.3, %159)
        参数含义:
        %val_if_large0.3 (Tensor): 输出，对比后的结果。
        %val_if_large.3 (Tensor): 输入，需要对比的Tensor1。
        %159 (Tensor): 输入，需要对比的Tensor2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    input_type = list(node.inputs())[1].type()
    if str(input_type) == "Tensor":
        # 处理输入0，即%val_if_large.3
        mapper._check_input(graph, inputs_node[0], inputs_name[0],
                            current_outputs, scope_name)
        layer_inputs["x"] = inputs_name[0]
        # 处理输入1，即%159
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["y"] = inputs_name[1]
        # 获取当前节点输入的list
        current_inputs = list(layer_inputs.values())
        graph.add_layer(
            "paddle.maximum",
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name)
    else:
        pass
    return current_inputs, current_outputs


def aten_max_pool2d(mapper, graph, node):
    """ 构造最大池化的PaddleLayer。
    TorchScript示例:
        %input.8 : Tensor = aten::max_pool2d(%result.11, %20, %23, %21, %22, %19)
        参数含义:
        %input.8 (Tensor): 输出，池化后的结果。
        %result.11 (Tensor): 需要池化的Tensor。
        %20 (list): 池化kernel的大小。
        %23 (list): 步长大小。
        %21 (list): 填充大小。
        %22 (list): 膨胀系数大小。
        %19 (bool): 是否用ceil函数计算输出高度和宽度。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("pool2d", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    layer_attrs_tmp = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.11
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%20
    layer_attrs["kernel_size"] = mapper.attrs[inputs_name[1]]
    layer_attrs_tmp["pool_size"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%23
    layer_attrs["stride"] = mapper.attrs[inputs_name[2]]
    layer_attrs_tmp["pool_stride"] = mapper.attrs[inputs_name[2]]
    # 处理输入3，即%21
    layer_attrs["padding"] = mapper.attrs[inputs_name[3]]
    layer_attrs_tmp["pool_padding"] = mapper.attrs[inputs_name[3]]
    # 处理输入4，即%22
    graph.add_layer(
        "prim.assert",
        inputs={},
        outputs=[inputs_name[4] + "_assert"],
        scope_name=scope_name + "_assert",
        type="eq",
        key=mapper.attrs[inputs_name[4]],
        value=[1, [1, 1]])
    # 处理输入5，即%19
    layer_attrs["ceil_mode"] = mapper.attrs[inputs_name[5]]
    layer_attrs_tmp["ceil_mode"] = mapper.attrs[inputs_name[5]]

    graph.add_layer(
        "paddle.nn.MaxPool2D",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_matmul(mapper, graph, node):
    """ 构造矩阵相乘的PaddleLayer。
    TorchScript示例:
        %output.2 : Tensor = aten::matmul(%101, %111)
        参数含义:
        %output.2 (Tensor): 输出，相乘后的结果。
        %101 (Tensor): 矩阵1。
        %102 (Tensor): 矩阵2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%101
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%102
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.matmul",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_min(mapper, graph, node):
    """ 构造获取最小值的PaddleLayer。
    TorchScript示例:
        %val_if_large0.3 : Tensor = aten::min(%val_if_large.3, %159)
        参数含义:
        %val_if_large0.3 (Tensor): 输出，对比后的结果。
        %val_if_large.3 (Tensor): 输入，需要对比的Tensor1。
        %159 (Tensor): 输入，需要对比的Tensor2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    input_type = list(node.inputs())[1].type()
    if str(input_type) == "Tensor":
        # 处理输入0，即%val_if_large.3
        mapper._check_input(graph, inputs_node[0], inputs_name[0],
                            current_outputs, scope_name)
        layer_inputs["x"] = inputs_name[0]
        # 处理输入1，即%159
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["y"] = inputs_name[1]
        # 获取当前节点输入的list
        current_inputs = list(layer_inputs.values())
        graph.add_layer(
            "paddle.minimum",
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name)
    else:
        pass
    return current_inputs, current_outputs


def aten_mean(mapper, graph, node):
    """ 构造求均值的PaddleLayer。
    TorchScript示例:
        %x.28 : Tensor = aten::mean(%result.1, %4967, %3, %2)
        参数含义:
        %x.28 (Tensor): 输出，求均值后的结果。
        %result.1 (Tensor): 输入，需要求均值的Tensor。
        %4967 (int/list): 求平均值运算的维度。
        %3 (bool): 是否在输出Tensor中保留减小的维度。
        %2 (Tensor): 结果Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%4967
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["axis"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入2，即%3
    if inputs_name[1] in mapper.attrs:
        layer_attrs["keepdim"] = mapper.attrs[inputs_name[2]]
    else:
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs, scope_name)
        layer_inputs["keepdim"] = inputs_name[2]
        current_inputs.append(inputs_name[2])

    graph.add_layer(
        "paddle.mean",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_meshgrid(mapper, graph, node):
    """ 构造对每个张量做扩充操作的PaddleLayer。
    TorchScript示例:
        %out.39 : int = aten::mshgrid(%input.1)
        参数含义:
        %out.39 (Tensor): 输出，扩充后的结果。
        %input.1 (Tensor): 输入。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["args"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = layer_inputs.values()
    current_outputs = layer_outputs

    graph.add_layer(
        "paddle.meshgrid",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_mul(mapper, graph, node):
    """ 构造数值相乘的PaddleLayer。
    TorchScript示例:
        %size_prods.39 : int = aten::mul(%size_prods.38, %114)
        参数含义:
        %size_prods.39 (Tensor): 输出，相乘后的结果。
        %size_prods.38 (-): 数值1。
        %114 (-): 数值2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%size_prods.38
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%114
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    current_outputs = layer_outputs

    graph.add_layer(
        "prim.mul",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_mul_(mapper, graph, node):
    """ 构造数值相乘的PaddleLayer。
    TorchScript示例:
        %size_prods.39 : int = aten::mul_(%size_prods.38, %114)
        参数含义:
        %size_prods.39 (Tensor): 输出，相乘后的结果。
        %size_prods.38 (-): 数值1。
        %114 (-): 数值2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%size_prods.38
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%114
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    current_outputs = layer_outputs

    graph.add_layer(
        "prim.mul",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_ne(mapper, graph, node):
    """ 构造判断数值是否不相等的PaddleLayer。
    TorchScript示例:
        %134 : bool = aten::ne(%133, %132)
        参数含义:
        %134 (bool): 对比后结果。
        %133 (-): 需对比的输入1。
        %132 (-): 需对比的输入2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%124
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%123
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.ne",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_neg(mapper, graph, node):
    """ 构造对数值取负的PaddleLayer。
    TorchScript示例:
        %909 : int = aten::neg(%908)
        参数含义:
        %909 (int): 取负后结果。
        %908 (int): 需取负的输入。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%124
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.neg",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_norm(mapper, graph, node):
    """ 构造计算范数的PaddleLayer。
    TorchScript示例:
        %25 = aten::norm(%input, %21, %58, %24)
        参数含义:
        %25 (Tensor): 取范数后的结果。
        %input (Tensor): 输入。
        %21 (int): 范数的种类。
        %58 (int): 使用范数计算的轴。
        %24 (bool): 是否在输出的Tensor中保留和输入一样的维度。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%21
    if inputs_name[1] in mapper.attrs:
        layer_attrs["p"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["p"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入2，即%58
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[2]]
    else:
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs, scope_name)
        layer_inputs["axis"] = inputs_name[2]
        current_inputs.append(inputs_name[2])
    # 处理输入3，即%24
    if inputs_name[1] in mapper.attrs:
        layer_attrs["keepdim"] = mapper.attrs[inputs_name[3]]
    else:
        mapper._check_input(graph, inputs_node[3], inputs_name[3],
                            current_outputs, scope_name)
        layer_inputs["keepdim"] = inputs_name[3]
        current_inputs.append(inputs_name[3])

    graph.add_layer(
        "paddle.norm",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten___not__(mapper, graph, node):
    """ 构造对bool型取负的PaddleLayer。
    TorchScript示例:
        %4498 : bool = aten::__not__(%aux_defined.2)
        参数含义:
        %4498 (bool): 取负后结果。
        %aux_defined.2 (bool): 需取负的输入。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%124
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.not",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_ones(mapper, graph, node):
    """ 构造创建固定形状、数据类型且值全为0的Tensor的PaddleLayer。
    TorchScript示例:
        %input.49 : Tensor = aten::ones(%23, %8, %6, %24, %5)
        参数含义:
        %input.49 (Tensor): 输出，全0的Tensor。
        %23 (list): 形状。
        %8 (int): 类型dtype。
        %6 (int): layout。
        %4995 (Device): 设备。
        %4995 (bool): 是否计算梯度。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    current_inputs = []
    # 处理输入0，即%23，代表end
    if inputs_name[0] in mapper.attrs:
        layer_attrs["shape"] = mapper.attrs[inputs_name[0]]
    else:
        mapper._check_input(graph, inputs_node[0], inputs_name[0],
                            current_outputs, scope_name)
        layer_inputs["shape"] = inputs_name[0]
        current_inputs.append(inputs_name[0])
    # 处理输入1，即%8，代表dtype
    layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[1]]]

    graph.add_layer(
        "paddle.ones",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_permute(mapper, graph, node):
    """ 构造对bool型取负的PaddleLayer。
    TorchScript示例:
        %2385 : Tensor = aten::permute(%cls_confs0.2, %2384)
        参数含义:
        %2385 (Tensor): 重排后的结果。
        %cls_confs0.2 (Tensor): 需要重排的Tensor。
        %2348 (list): 依照此参数进行重排。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%cls_confs0.2
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%2348
    if inputs_name[1] in mapper.attrs:
        layer_attrs["perm"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["perm"] = inputs_name[1]
        current_inputs.append(inputs_name[1])

    graph.add_layer(
        "paddle.transpose",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_pixel_shuffle(mapper, graph, node):
    """ 构造以像素的方式重排的PaddleLayer。
    TorchScript示例:
        %x.6 : aten::pixel_shuffle(%input.101, %726)
        参数含义:
        %x.6 (Tensor): 输出，重排后的Tensor。
        %input.101 (Tensor): 需要重排的Tensor。
        %726 (int): 增大空间分辨率的增大因子。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.101
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%726
    layer_attrs["upscale_factor"] = mapper.attrs[inputs_name[1]]

    graph.add_layer(
        "paddle.nn.functional.pixel_shuffle",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_pow(mapper, graph, node):
    """ 构造指数激活的PaddleLayer。
    TorchScript示例:
        %x.6 : Tensor = aten::pow(%4700, %4703)
        参数含义:
        %x.6 (Tensor): 输出，指数激活后的Tensor。
        %4700 (Tensor): 需要指数激活的Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%4700
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%4703
    if inputs_name[1] in mapper.attrs:
        layer_attrs["y"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["y"] = inputs_name[1]
        current_inputs.append(inputs_name[1])

    graph.add_layer(
        "paddle.pow",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_prelu(mapper, graph, node):
    """ 构造prelu激活的PaddleLayer。
    TorchScript示例:
        %result.3 : aten::prelu(%input.150, %999)
        参数含义:
        %result.3 (Tensor): 输出，prelu后的结果。
        %input.150 (Tensor): 需要prelu的Tensor。
        %999 (Tnsor): 权重。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("relu", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.150
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%999
    weight = mapper.pytorch_params[inputs_name[1]]
    mapper.paddle_params[op_name + "._weight"] = weight
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.nn.PReLU",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        num_parameters=weight.shape[0])
    return current_inputs, current_outputs


def aten_reflection_pad1d(mapper, graph, node):
    """ 构造1维映射填充的PaddleLayer。
    TorchScript示例:
        %6 = aten::reflection_pad1d(%input, %7)
        参数含义:
        %6 (Tensor): 输出，填充后的Tensor。
        %input (Tensor): 需要填充的Tensor。
        %7 (list|Tensor): 填充大小。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("pad1d", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%7
    if inputs_name[1] in mapper.attrs:
        layer_attrs["padding"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        ipt_node = inputs_node[1]
        while ipt_node.kind() != "prim::GetAttr":
            inputs_name, inputs_node = mapper._get_inputs_name(ipt_node)
            ipt_node = inputs_node[0]
        layer_attrs["padding"] = list(mapper.pytorch_params[inputs_name[0]])
    layer_attrs["mode"] = string("reflect")

    graph.add_layer(
        "paddle.nn.Pad1D",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_reflection_pad2d(mapper, graph, node):
    """ 构造2维映射填充的PaddleLayer。
    TorchScript示例:
        %6 = aten::reflection_pad2d(%input, %7)
        参数含义:
        %6 (Tensor): 输出，填充后的Tensor。
        %input (Tensor): 需要填充的Tensor。
        %7 (list|Tensor): 填充大小。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("pad2d", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%7
    if inputs_name[1] in mapper.attrs:
        layer_attrs["padding"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        ipt_node = inputs_node[1]
        while ipt_node.kind() != "prim::GetAttr":
            inputs_name, inputs_node = mapper._get_inputs_name(ipt_node)
            ipt_node = inputs_node[0]
        layer_attrs["padding"] = list(mapper.pytorch_params[inputs_name[0]])
    layer_attrs["mode"] = string("reflect")

    graph.add_layer(
        "paddle.nn.Pad2D",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_relu(mapper, graph, node):
    """ 构造ReLU激活的PaddleLayer。
    TorchScript示例:
        %result.3 : Tensor = aten::relu(%input.5)
        参数含义:
        %result.3 (Tensor): 输出，ReLU后的结果。
        %result.5 (Tensor): 需要ReLU的Tensor。
    注意: inplace这个参数在paddle中未实现
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("relu", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.nn.ReLU",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_relu_(mapper, graph, node):
    """ 构造ReLU激活的PaddleLayer。
    TorchScript示例:
        %result.3 : Tensor = aten::relu_(%input.5)
        参数含义:
        %result.3 (Tensor): 输出，ReLU后的结果。
        %result.5 (Tensor): 需要ReLU的Tensor。
    注意: inplace这个参数在paddle中未实现
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("relu", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.nn.ReLU",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_relu6(mapper, graph, node):
    """ 构造ReLU6激活的PaddleLayer。
    TorchScript示例:
        %result.3 : Tensor = aten::relu6(%input.5)
        参数含义:
        %result.3 (Tensor): 输出，ReLU6后的结果。
        %result.5 (Tensor): 需要ReLU6的Tensor。
    注意: inplace这个参数在paddle中未实现
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("relu6", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.nn.ReLU6",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_remainder(mapper, graph, node):
    """ 构造取余数的PaddleLayer。
    TorchScript示例:
        %701 : Tensor = aten::remainder(%661, %139)
        参数含义:
        %701 (Tensor): 输出，取余结果的Tensor。
        %661 (Tensor): 需要取余的Tensor。
        %139 (Tensor): 除数Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%661
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%139
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.remainder",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_repeat(mapper, graph, node):
    """ 构造根据参数对输入各维度进行复制的PaddleLayer。
    TorchScript示例:
        %701 : Tensor = aten::repeat(%699, %700)
        参数含义:
        %701 (Tensor): 输出，复制后的Tensor。
        %699 (Tensor): 需要复制的Tensor。
        %700 (list): 指定每个维度复制的次数。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%699
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%700
    if inputs_name[1] in mapper.attrs:
        layer_attrs["repeat_times"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["repeat_times"] = inputs_name[1]
        current_inputs.append(inputs_name[1])

    graph.add_layer(
        "paddle.tile",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_reshape(mapper, graph, node):
    """ 构造调整大小的PaddleLayer。
    TorchScript示例:
        %x.6 : Tensor = aten::reshape(%4700, %4703)
        参数含义:
        %x.6 (Tensor): 输出，reshape后的Tensor。
        %4700 (Tensor): 需要reshape的Tensor。
        %4703 (list): 形状大小组成的list。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%4700
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%4703
    if inputs_name[1] in mapper.attrs:
        layer_attrs["shape"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["shape"] = inputs_name[1]
        current_inputs.append(inputs_name[1])

    graph.add_layer(
        "paddle.reshape",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_roll(mapper, graph, node):
    """ 构造循环滚动的PaddleLayer。
    TorchScript示例:
        %x.87 : Float = aten::roll(%x.86, %1862, %1863)
        参数含义:
        %x.87 (Tensor): 输出Tensor。
        %x.86 (Tensor): 输入Tensor。
        %1862 (int/list/tuple): 滚动位移。
        %1863 (int/list/tuple): 滚动轴。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.86
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%1862
    if inputs_name[1] in mapper.attrs:
        layer_attrs["shifts"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["shifts"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入2，即%1863
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[2]]
    else:
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs, scope_name)
        layer_inputs["axis"] = inputs_name[2]
        current_inputs.append(inputs_name[2])

    graph.add_layer(
        "paddle.roll",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_rsub(mapper, graph, node):
    """ 构造数值相减的PaddleLayer，计算公式为：out = y - alpha * x。
    TorchScript示例:
        %31 : Tensor = aten::rsub(%30, %13, %7)
        参数含义:
        %31 (Tensor): 相减结果。
        %30 (Tensor): 输入Tensor x。
        %13 (int/float): 输入数值 y。
        %7 (int/float): alpha。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%30
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%13
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 处理输入2，即%7
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    layer_inputs["alpha"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.rsub",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_ScalarImplicit(mapper, graph, node):
    """ 构造获取scalar的PaddleLayer。
    TorchScript示例:
        %89 : Scalar = aten::ScalarImplicit(%end.1)
        参数含义:
        %89 (Scalar): 输出，得到的Scalar。
        %end.1 (-): 组要转换的数据。
    【注意】由于Paddle无Scalar，所以最后转换为Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%end.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    input_type = list(node.inputs())[0].type()
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    if str(input_type) == "Tensor":
        graph.add_layer(
            "prim.equal",
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name)
    else:
        raise Exception(
            "The input type {} of aten::ScalarImplicit is not implemented yet!"
        ).format(input_type)
    return current_inputs, current_outputs


def aten_select(mapper, graph, node):
    """ 构造选取特定维度Variable的PaddleLayer。
    TorchScript示例:
        %19 : Tensor = aten::select(%18, %8, %7)
        参数含义:
        %19 (Tensor): 输出，选取的Tensor。
        %18 (Tensor): 需要选取的Tensor。
        %8 (int): select的维度。
        %7 (int): select的第n个向量。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%18
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 处理输入1，即%8
    layer_attrs["dim"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%75
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    layer_inputs["index"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.select",
        inputs=layer_inputs,
        outputs=current_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten__set_item(mapper, graph, node):
    """ 构造对dict加入元素的PaddleLayer。
    TorchScript示例:
        = aten::_set_item(%features.1, %out_name.1, %x.3)
        参数含义:
        %features.1 (list): dict。
        %out_name.1 (-): dict的key。
        %x.3 (-): dict的value。
    """
    scope_name = mapper.normalize_scope_name(node)
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = []
    # 处理输入0，即%features.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["dict"] = inputs_name[0]
    # 处理输入1，即%out_name.1
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["key"] = inputs_name[1]
    # 处理输入2，即%x.3
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    layer_inputs["value"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.set_item", inputs=layer_inputs, outputs=[], scope_name=scope_name)
    return current_inputs, current_outputs


def aten_sigmoid(mapper, graph, node):
    """ 构造sigmoid激活的PaddleLayer。
    TorchScript示例:
        %55 : Tensor = aten::sigmoid(%54)
        参数含义:
        %55 (Tensor): 输出，sigmoid后的结果。
        %54 (Tensor): 需要tanh的Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("sigmoid", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%54
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.nn.Sigmoid",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_sin(mapper, graph, node):
    """ 构造数学计算sin的PaddleLayer。
    TorchScript示例:
        %94 : Tensor = aten::sin(%sinusoid_inp.1)
        参数含义:
        %94 (Tensor): 输出，sin之后的结果。
        %sinusoid_inp.1 (Tensor): 需要进行shape的Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%sinusoid_inp.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.sin",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_size(mapper, graph, node):
    """ 构造获取shape的PaddleLayer。
    TorchScript示例:
        %73 : int[] = aten::size(%x.12, %10)
        参数含义:
        %73 (list): 输出，shape的list。
        %x.12 (Tensor): 需要获取shape的Tensor。
        %10 (int): 非必须，代表维度。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.12
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    if len(inputs_name) > 1:
        # 处理输入1，即%12
        if inputs_name[1] in mapper.attrs:
            layer_attrs["dim"] = mapper.attrs[inputs_name[1]]
        else:
            mapper._check_input(graph, inputs_node[1], inputs_name[1],
                                current_outputs, scope_name)
            layer_inputs["dim"] = inputs_name[1]
            current_inputs.append(inputs_name[1])
        graph.add_layer(
            "prim.shape_dim",
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name,
            **layer_attrs)
        return current_inputs, current_outputs

    graph.add_layer(
        "prim.shape",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_slice(mapper, graph, node):
    """ 构造切分list或Variable的PaddleLayer。
    TorchScript示例:
        %83 : int[] = aten::slice(%73, %_81, %82, %75, %77)
        参数含义:
        %83 (list/Tensor): 输出，切分后的list。
        %73 (list/Tensor): 需要切分的list。
        %_81 (int): 切分的维度，不一定存在。
        %82 (int): 切分的开始索引。
        %75 (int): 切分的结束索引。
        %77 (int): 切分的步长。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    if len(inputs_name) == 5:
        # 处理输入0，即%73
        mapper._check_input(graph, inputs_node[0], inputs_name[0],
                            current_outputs, scope_name)
        layer_inputs["x"] = inputs_name[0]

        # 获取当前节点输入的list
        current_inputs = list(layer_inputs.values())
        # 处理输入1，即%_81
        if inputs_name[1] in mapper.attrs:
            graph.add_layer(
                "prim.list",
                inputs={},
                outputs=[inputs_name[1] + "_list"],
                scope_name=scope_name,
                input0=mapper.attrs[inputs_name[1]])
        else:
            mapper._check_input(graph, inputs_node[1], inputs_name[1],
                                current_outputs, scope_name)
            graph.add_layer(
                "prim.list",
                inputs={"input0": inputs_name[1]},
                outputs=[inputs_name[1] + "_list"],
                scope_name=scope_name)
            current_inputs.append(inputs_name[1])
        layer_inputs["axes"] = inputs_name[1] + "_list"
        current_inputs.append(inputs_name[1] + "_list")
        current_outputs.append(inputs_name[1] + "_list")
        # 处理输入2，即%82
        if inputs_name[2] in mapper.attrs:
            graph.add_layer(
                "prim.list",
                inputs={},
                outputs=[inputs_name[2] + "_list"],
                scope_name=scope_name,
                input0=mapper.attrs[inputs_name[2]])
        else:
            mapper._check_input(graph, inputs_node[2], inputs_name[2],
                                current_outputs, scope_name)
            graph.add_layer(
                "prim.list",
                inputs={"input0": inputs_name[2]},
                outputs=[inputs_name[2] + "_list"],
                scope_name=scope_name)
            current_inputs.append(inputs_name[2])
        layer_inputs["starts"] = inputs_name[2] + "_list"
        current_inputs.append(inputs_name[2] + "_list")
        current_outputs.append(inputs_name[2] + "_list")
        # 处理输入3，即%85
        if inputs_name[3] in mapper.attrs:
            graph.add_layer(
                "prim.list",
                inputs={},
                outputs=[inputs_name[3] + "_list"],
                scope_name=scope_name,
                input0=mapper.attrs[inputs_name[3]])
        else:
            mapper._check_input(graph, inputs_node[3], inputs_name[3],
                                current_outputs, scope_name)
            graph.add_layer(
                "prim.list",
                inputs={"input0": inputs_name[3]},
                outputs=[inputs_name[3] + "_list"],
                scope_name=scope_name)
            current_inputs.append(inputs_name[3])
        layer_inputs["ends"] = inputs_name[3] + "_list"
        current_inputs.append(inputs_name[3] + "_list")
        current_outputs.append(inputs_name[3] + "_list")
        # 处理输入4，即%77
        if inputs_name[4] in mapper.attrs:
            graph.add_layer(
                "prim.list",
                inputs={},
                outputs=[inputs_name[4] + "_list"],
                scope_name=scope_name,
                input0=mapper.attrs[inputs_name[4]])
        else:
            mapper._check_input(graph, inputs_node[4], inputs_name[4],
                                current_outputs, scope_name)
            graph.add_layer(
                "prim.list",
                inputs={"input0": inputs_name[4]},
                outputs=[inputs_name[4] + "_list"],
                scope_name=scope_name)
            current_inputs.append(inputs_name[4])
        layer_inputs["strides"] = inputs_name[4] + "_list"
        current_inputs.append(inputs_name[4] + "_list")
        current_outputs.append(inputs_name[4] + "_list")

        graph.add_layer(
            "paddle.strided_slice",
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name)
    else:
        # 处理输入0，即%73
        mapper._check_input(graph, inputs_node[0], inputs_name[0],
                            current_outputs, scope_name)
        layer_inputs["input"] = inputs_name[0]
        # 处理输入1，即%82
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["start"] = inputs_name[1]
        # 处理输入2，即%75
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs, scope_name)
        layer_inputs["end"] = inputs_name[2]
        # 处理输入3，即%77
        mapper._check_input(graph, inputs_node[3], inputs_name[3],
                            current_outputs, scope_name)
        layer_inputs["step"] = inputs_name[3]
        # 获取当前节点输入的list
        current_inputs = list(layer_inputs.values())

        graph.add_layer(
            "prim.slice",
            inputs=layer_inputs,
            outputs=layer_outputs,
            scope_name=scope_name)
    return current_inputs, current_outputs


def aten_softmax(mapper, graph, node):
    """ 构造softmax激活的PaddleLayer。
    TorchScript示例:
        %input2.1 : Tensor = aten::softmax(%input.5, %80, %72)
        参数含义:
        %input2.1 (Tensor): 激活后结果。
        %input.5 (Tensor): 需要激活的Tensor。
        %80 (int): 指定对输入Tensor进行运算的轴。
        %72 (str): 类型，默认为None。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("softmax", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.31
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    layer_attrs["axis"] = mapper.attrs[inputs_name[1]]

    graph.add_layer(
        "paddle.nn.Softmax",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_softplus(mapper, graph, node):
    """ 构造softplus激活的PaddleLayer。
    TorchScript示例:
        %54 : Tensor = aten::softplus(%x.31, %30, %29)
        参数含义:
        %54 (Tensor): 激活后结果。
        %x.31 (Tensor): 需要激活的Tensor。
        %30 (int): beta。
        %29 (int): 阈值。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("softplus", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.31
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    layer_attrs["beta"] = mapper.attrs[inputs_name[1]]
    layer_attrs["threshold"] = mapper.attrs[inputs_name[2]]

    graph.add_layer(
        "paddle.nn.Softplus",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_split_with_sizes(mapper, graph, node):
    """ 构构造split的PaddleLayer。
    TorchScript示例:
        %1450 : Tensor[] = aten::split_with_sizes(%1446, %1750, %41)
        参数含义:
        %1450 (Tensor): 输出，split后的Tensor。
        %1446 (Tensor): 需要获取split的Tensor。
        %1750 (list): 子Tensor的数量列表。
        %41 (int): 需要分割的维度。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%1446
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%1750
    if inputs_name[1] in mapper.attrs:
        layer_attrs["num_or_sections"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["num_or_sections"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入2，即%135
    if inputs_name[2] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[2]]
    else:
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs, scope_name)
        layer_inputs["axis"] = inputs_name[2]
        current_inputs.append(inputs_name[2])
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.split",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_sqrt(mapper, graph, node):
    """ 构构造sqrt的PaddleLayer。
    TorchScript示例:
        %787 : Tensor = aten::sqrt(%786)
        参数含义:
        %787 (Tensor): 输出，取sqrt的Tensor。
        %786 (Tensor): 需要获取sqrt的Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%786
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.sqrt",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_squeeze(mapper, graph, node):
    """ 构造删除位数为1的维度的PaddleLayer。
    TorchScript示例:
        %12 : Tensor = aten::squeeze(%start_logits.1, %4)
        参数含义:
        %12 (Tensor): 输出，删除维度后的Tensor。
        %start_logits.1 (Tensor): 需要删除维度的Tensor。
        %4 (int): 维度。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%start_logits.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%4
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["axis"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    graph.add_layer(
        "paddle.squeeze",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_stack(mapper, graph, node):
    """ 构造堆叠Tensor的PaddleLayer。
    TorchScript示例:
        %x.222 : Tensor = aten::stack(%32, %7)
        参数含义:
        %x.222 (Tensor): 输出，堆叠后的结果。
        %i.12 (Tensor): 需要堆叠的Tensor组成的Tensor。
        %7 (int): 堆叠的轴。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%13
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%12
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["axis"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    graph.add_layer(
        "paddle.stack",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_sub(mapper, graph, node):
    """ 构造数值相减的PaddleLayer。
    TorchScript示例:
        %840 : int = aten::sub(%839, %836, %3)
        参数含义:
        %840 (-): 相减结果。
        %839 (-): 输入数值 x。
        %836 (-): 输入数值 y。
        %3 (-): alpha。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%839
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%836
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[1]
    # 处理输入2，即%3
    if len(inputs_node) > 2:
        if inputs_name[2] in mapper.attrs:
            layer_attrs["alpha"] = mapper.attrs[inputs_name[2]]
        else:
            mapper._check_input(graph, inputs_node[2], inputs_name[2],
                                current_outputs, scope_name)
            layer_inputs["alpha"] = inputs_name[2]
            current_inputs.append(inputs_name[2])
    else:
        layer_attrs["alpha"] = 1.0
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.sub",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_sub_(mapper, graph, node):
    """ 构造数值相减的PaddleLayer。
    TorchScript示例:
        %840 : int = aten::sub_(%839, %836, %3)
        参数含义:
        %840 (-): 相减结果。
        %839 (-): 输入数值 x。
        %836 (-): 输入数值 y。
        %3 (-): alpha。
    """
    return aten_sub(mapper, graph, node)


def aten_t(mapper, graph, node):
    """ 构造矩阵转置的PaddleLayer。
    TorchScript示例:
        %840 : int = aten::sub(%839, %836)
        参数含义:
        %109 (Tensor): 输出，转置后的矩阵。
        %102 (Tensor): 需要转置的Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.12
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.transpose",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        perm=[1, 0])
    return current_inputs, current_outputs


def aten_tanh(mapper, graph, node):
    """ 构造tanh激活的PaddleLayer。
    TorchScript示例:
        %55 : Tensor = aten::tanh(%54)
        参数含义:
        %55 (Tensor): 输出，tanh后的结果。
        %54 (Tensor): 需要tanh的Tensor。
    """
    scope_name = mapper.normalize_scope_name(node)
    op_name = name_generator("tanh", mapper.nn_name2id)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [op_name, output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.nn.Tanh",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_split(mapper, graph, node):
    """ 构造分割Tensor的PaddleLayer。
    TorchScript示例:
        %160 : Tensor[] = aten::split(%159, %135, %123)
        参数含义:
        %160 (Tensor): 输出，分割后的矩阵。
        %159 (Tensor): 需要分割的Tensor。
        %135 (int): 分割的数量。
        %123 (int): 轴。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%159
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入2，即%723
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    layer_inputs["axis"] = inputs_name[2]
    # 处理输入1，即%135
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    input_type = list(node.inputs())[0].type()
    if "[]" in str(input_type):
        layer_inputs["num_or_sections"] = inputs_name[1]
    else:
        index = mapper.attrs[inputs_name[2]]
        graph.add_layer(
            "prim.shape",
            inputs={"input": inputs_name[0]},
            outputs=[inputs_name[0] + '_shape'],
            scope_name=scope_name)
        graph.add_layer(
            "prim.getitem",
            inputs={"list": inputs_name[0] + '_shape'},
            outputs=[inputs_name[0] + '_dim'],
            scope_name=scope_name,
            index=index)
        graph.add_layer(
            "prim.floordiv",
            inputs={'x': inputs_name[0] + '_dim',
                    'y': inputs_name[1]},
            outputs=[inputs_name[1] + '_div'],
            scope_name=scope_name)
        layer_attrs["num_or_sections"] = inputs_name[1] + '_div'
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.split",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_transpose(mapper, graph, node):
    """ 构造矩阵转置的PaddleLayer。
    TorchScript示例:
        %715 : Tensor = aten::transpose(%x.21, %704, %705)
        参数含义:
        %715 (Tensor): 输出，转置后的矩阵。
        %x.21 (Tensor): 需要转置的Tensor。
        %704 (int): 转置的维度1。
        %705 (int): 转置的维度2。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.21
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%704
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    dim1 = inputs_name[1]
    # 处理输入2，即%705
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    dim2 = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    graph.add_layer(
        "prim.shape",
        inputs={"input": inputs_name[0]},
        outputs=[output_name + "_shape"],
        scope_name=scope_name)
    current_outputs.append(output_name + "_shape")
    graph.add_layer(
        "prim.len",
        inputs={"input": output_name + "_shape"},
        outputs=[output_name + "_len"],
        scope_name=scope_name)
    current_outputs.append(output_name + "_len")
    current_inputs.append(output_name + "_shape")
    graph.add_layer(
        "prim.len2list",
        inputs={"len": output_name + "_len"},
        outputs=[output_name + "_list"],
        scope_name=scope_name)
    current_outputs.append(output_name + "_list")
    current_inputs.append(output_name + "_len")
    graph.add_layer(
        "prim.check_dim",
        inputs={"len": output_name + "_len",
                "dim": dim1},
        outputs=[dim1 + "_new"],
        scope_name=scope_name)
    graph.add_layer(
        "prim.check_dim",
        inputs={"len": output_name + "_len",
                "dim": dim2},
        outputs=[dim2 + "_new"],
        scope_name=scope_name)
    graph.add_layer(
        "prim.replaceitem",
        inputs={
            "list": output_name + "_list",
            "index": dim1 + "_new",
            "item": dim2 + "_new"
        },
        outputs=[],
        scope_name=scope_name)
    graph.add_layer(
        "prim.replaceitem",
        inputs={
            "list": output_name + "_list",
            "index": dim2 + "_new",
            "item": dim1 + "_new"
        },
        outputs=[],
        scope_name=scope_name)
    graph.add_layer(
        "paddle.transpose",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        perm=output_name + "_list")
    return current_inputs, current_outputs


def aten_to(mapper, graph, node):
    """ 构造类型转换的PaddleLayer。
    TorchScript示例:
        %30 : Tensor = aten::to(%extended_attention_mask.1, %12, %5, %5, %4)
        参数含义:
        %30 (Tensor): 转换后的Tensor。
        %extended_attention_mask.1 (Tensor): 需要转换的Tensor。
        %12 (int): 转换的类型。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%13
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%12
    if len(inputs_name) == 6:
        layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[2]]]
    else:
        layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[1]]]

    graph.add_layer(
        "paddle.cast",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_type_as(mapper, graph, node):
    """ 构造转换Tensor类型的PaddleLayer。
    TorchScript示例:
        %57 : Tensor = aten::type_as(%56, %mask.1)
        参数含义:
        %57 (Tensor): 输出，改变类型后的Tensor。
        %56 (Tensor): 需要改变类型的Tensor。
        %mask.1 (Tensor): 转换成与该Tensor相一致的类型。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%56
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入0，即%mask.1
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    graph.add_layer(
        "prim.type",
        inputs={"input": inputs_name[1]},
        outputs=[inputs_name[1] + "_type"],
        scope_name=scope_name)
    layer_inputs["dtype"] = inputs_name[1] + "_type"
    current_inputs.append(inputs_name[1])

    graph.add_layer(
        "paddle.cast",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_unsqueeze(mapper, graph, node):
    """ 构造插入维度的PaddleLayer。
    TorchScript示例:
        %13 : Tensor = aten::unsqueeze(%12, %7)
        参数含义:
        %13 (Tensor): 输出，插入维度后的Tensor。
        %12 (Tensor): 需要插入维度的Tensor。
        %7 (int): 维度。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%13
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%12
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["axis"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    graph.add_layer(
        "paddle.unsqueeze",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_upsample_bilinear2d(mapper, graph, node):
    """ 构造使用bilinear上采样的PaddleLayer。
    TorchScript示例:
        %4997 : Tensor = aten::upsample_bilinear2d(%x.13, %4963, %5421, %4995, %4996)
        参数含义:
        %4997 (Tensor): 输出，上采样后的Tensor。
        %x.13 (Tensor): 需要上采样的Tensor。
        %4963 (list): 上采样后的大小。
        %5421 (bool): 若为True，则将输入和输出张量的4个角落像素的中心对齐，并保留角点像素的值。
        %4995 (float): 高度的乘数因子。
        %4996 (float): 宽度的乘数因子。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.13
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%4963
    if inputs_name[1] in mapper.attrs:
        layer_attrs["size"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["size"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
        graph.add_layer(
            "prim.isinstance",
            inputs={"input": inputs_name[1]},
            outputs=[inputs_name[1] + "_isinstance"],
            scope_name=scope_name,
            cls="paddle.fluid.Variable")
        # TODO(syf): paddle.Variable
        graph.add_layer(
            "prim.if", {"input": inputs_name[1] + "_isinstance"},
            outputs=[inputs_name[0] + "_if1"],
            scope_name=scope_name)
        if_layer = graph.layers[list(graph.layers.keys())[-1]]
        block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
        block.add_layer(
            "prim.var2list",
            inputs={"input": inputs_name[1]},
            outputs=[inputs_name[1]],
            scope_name=scope_name)
        if_layer.add_block(block)
        block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
        if_layer.add_block(block)
        if_layer.inputs["input-0"] = inputs_name[1]
    # 处理输入2，即%5421
    if inputs_name[2] in mapper.attrs:
        layer_attrs["align_corners"] = mapper.attrs[inputs_name[2]]
    else:
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs, scope_name)
        layer_inputs["align_corners"] = inputs_name[2]
        current_inputs.append(inputs_name[2])
    if "size" in layer_attrs and layer_attrs["size"] is None:
        mapper._check_input(graph, inputs_node[3], inputs_name[3],
                            current_outputs, scope_name)
        layer_inputs["scale_factor"] = inputs_name[3]
    layer_attrs["align_mode"] = 0
    layer_attrs["mode"] = string("bilinear")
    graph.add_layer(
        "paddle.nn.functional.interpolate",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_upsample_nearest2d(mapper, graph, node):
    """ 构造使用nearest上采样的PaddleLayer。
    TorchScript示例:
        %4997 : Tensor = aten::upsample_nearest2d(%x.13, %4963, %5421, %4995)
        参数含义:
        %4997 (Tensor): 输出，上采样后的Tensor。
        %x.13 (Tensor): 需要上采样的Tensor。
        %4963 (list): 上采样后的大小。
        %5421 (float): 高度的乘数因子。
        %4995 (float): 宽度的乘数因子。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.13
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%4963
    if inputs_name[1] in mapper.attrs:
        layer_attrs["size"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["size"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
        graph.add_layer(
            "prim.isinstance",
            inputs={"input": inputs_name[1]},
            outputs=[inputs_name[1] + "_isinstance"],
            scope_name=scope_name,
            cls="paddle.fluid.Variable")
        # TODO(syf): paddle.Variable
        graph.add_layer(
            "prim.if", {"input": inputs_name[1] + "_isinstance"},
            outputs=[inputs_name[0] + "_if1"],
            scope_name=scope_name)
        if_layer = graph.layers[list(graph.layers.keys())[-1]]
        block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
        block.add_layer(
            "prim.var2list",
            inputs={"input": inputs_name[1]},
            outputs=[inputs_name[1]],
            scope_name=scope_name)
        if_layer.add_block(block)
        block = PaddleGraph(source_type="pytorch", parent_layer=if_layer)
        if_layer.add_block(block)
        if_layer.inputs["input-0"] = inputs_name[1]
    if "size" in layer_attrs and layer_attrs["size"] is None:
        mapper._check_input(graph, inputs_node[3], inputs_name[3],
                            current_outputs, scope_name)
        layer_inputs["scale_factor"] = inputs_name[3]
    layer_attrs["align_mode"] = 0
    layer_attrs["mode"] = string("nearest")
    graph.add_layer(
        "paddle.nn.functional.interpolate",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_values(mapper, graph, node):
    """ 构造对比大小的PaddleLayer。
    TorchScript示例:
        %5 : Float(1, *, 1024, 2048)[] = aten::values(%1)
        参数含义:
        %5 (list): 输出，由字典获取的values的list。
        %1 (dict): 字典。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%78
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.dict2values",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_view(mapper, graph, node):
    """ 构造调整大小的PaddleLayer。
    TorchScript示例:
        %input.152 : Tensor = aten::view(%x.20, %430)
        参数含义:
        %input.152 (Tensor): 输出，view后的Tensor。
        %x.20 (Tensor): 需要view的Tensor。
        %430 (list): 形状大小组成的list。
    【注意】view 函数只能用于contiguous后的Tensor上，
          也就是只能用于内存中连续存储的Tensor。
          如果对Tensor调用过transpose，permute等操作的话会使该Tensor在内存中变得不再连续，
          此时就不能再调用view函数。因此，需要先使用contiguous来返回一个contiguous copy。
          reshape则不需要依赖目标Tensor是否在内存中是连续的。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.20
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%430
    if inputs_name[1] in mapper.attrs:
        layer_attrs["shape"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["shape"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    graph.add_layer(
        "paddle.reshape",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_warn(mapper, graph, node):
    """ 构造warning的PaddleLayer。
    TorchScript示例:
        = aten::warn(%3, %2)
        参数含义:
        %3 (str): warning的提示字符串。
        %2 (int): warning的stacklevel。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%3
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%2
    if inputs_name[1] in mapper.attrs:
        layer_attrs["stacklevel"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs, scope_name)
        layer_inputs["stacklevel"] = inputs_name[1]
        current_inputs.append(inputs_name[1])

    graph.add_layer(
        "prim.warnings",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_where(mapper, graph, node):
    """ 构造返回一个根据输入condition, 选择x或y的元素组成的多维Tensor的PaddleLayer，该节点实现out = x + y。
    TorchScript示例:
        %input.4 : Tensor = aten::where(%209, %w0.2, %210)
        参数含义:
        %input.4 (Tensor): 选择的结果。
        %209 (Tensor): 条件。
        %w0.2 (Tensor): 输入数值 x。
        %210 (Tensor): 输入数值 y。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%209
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["condition"] = inputs_name[0]
    # 处理输入1，即%w0.2
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[1]
    # 处理输入1，即%w0.2
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs,
                        scope_name)
    layer_inputs["y"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "paddle.where",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name)
    return current_inputs, current_outputs


def aten_zeros(mapper, graph, node):
    """ 构造创建固定形状、数据类型且值全为0的Tensor的PaddleLayer。
    TorchScript示例:
        %input.49 : Tensor = aten::zeros(%23, %8, %6, %24, %5)
        参数含义:
        %input.49 (Tensor): 输出，全0的Tensor。
        %23 (list): 形状。
        %8 (int): 类型dtype。
        %6 (int): layout。
        %4995 (Device): 设备。
        %4995 (bool): 是否计算梯度。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    current_inputs = []
    # 处理输入0，即%23，代表end
    if inputs_name[0] in mapper.attrs:
        layer_attrs["shape"] = mapper.attrs[inputs_name[0]]
    else:
        mapper._check_input(graph, inputs_node[0], inputs_name[0],
                            current_outputs, scope_name)
        layer_inputs["shape"] = inputs_name[0]
        current_inputs.append(inputs_name[0])
    # 处理输入1，即%8，代表dtype
    layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[1]]]

    graph.add_layer(
        "paddle.zeros",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_zeros_like(mapper, graph, node):
    """ 构造创建与输入Tensor形状一致的、数据类型且值全为0的Tensor的PaddleLayer。
    TorchScript示例:
        %782 : Tensor = aten::zeros_like(%n.2, %655, %670, %662, %671, %672)
        参数含义:
        %782 (Tensor): 输出，全0的Tensor。
        %n.2 (Tensor): 标准Tensor。
        %655 (int): 类型dtype。
        %670 (int): layout。
        %662 (Device): 设备。
        %671 (bool): 是否计算梯度。
        %672 (memory_format): 存储类型。
    """
    scope_name = mapper.normalize_scope_name(node)
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%n.2
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs,
                        scope_name)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%655，代表dtype
    layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[1]]]

    graph.add_layer(
        "paddle.zeros_like",
        inputs=layer_inputs,
        outputs=layer_outputs,
        scope_name=scope_name,
        **layer_attrs)
    return current_inputs, current_outputs
