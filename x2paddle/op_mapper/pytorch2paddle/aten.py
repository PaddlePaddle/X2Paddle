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

from x2paddle.core.util import *


def aten_adaptive_avg_pool2d(mapper, graph, node):
    """ 构造average adaptive pool2d的PaddleLayer。

    TorchScript示例:
        %x.5 : Tensor = aten::adaptive_avg_pool2d(%x.3, %_output_size.1)
        参数含义:
        %x.5 (Tensor): 池化后结果Tensor。
        %x.3 (Tensor): 输入Tensor。
        %_output_size.1 (list): 自适应池化后的Tensor的宽、高大小。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.3
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%_output_size.1
    if inputs_name[1] in mapper.attrs:
        layer_attrs["pool_size"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_attrs["pool_size"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    layer_attrs["pool_type"] = string("avg")

    graph.add_layer(
        "fluid.layers.adaptive_pool2d",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%150
    mapper._check_input(
        graph, inputs_node[0], inputs_name[0], current_outputs, add_dim=True)
    layer_inputs["input"] = inputs_name[0]
    # 处理输入1，即%input.3
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["x"] = inputs_name[1]
    # 处理输入2，即%156
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs)
    layer_inputs["y"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入3，即%152
    if inputs_name[3] in mapper.attrs:
        layer_attrs["beta"] = mapper.attrs[inputs_name[3]]
    else:
        mapper._check_input(graph, inputs_node[3], inputs_name[3],
                            current_outputs)
        layer_attrs["beta"] = inputs_name[3]
        current_inputs.append(inputs_name[3])
    # 处理输入4，即%151
    if inputs_name[4] in mapper.attrs:
        layer_attrs["alpha"] = mapper.attrs[inputs_name[4]]
    else:
        mapper._check_input(graph, inputs_node[4], inputs_name[4],
                            current_outputs)
        layer_attrs["alpha"] = inputs_name[4]
        current_inputs.append(inputs_name[4])

    graph.add_layer(
        "fluid.layers.addmm",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%i.12
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%288
    mapper._check_input(
        graph, inputs_node[1], inputs_name[1], current_outputs, add_dim=True)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.add", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%output.2
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%150
    mapper._check_input(
        graph, inputs_node[1], inputs_name[1], current_outputs, add_dim=True)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入2，即%151
    if inputs_name[2] in mapper.attrs:
        layer_attrs["alpha"] = mapper.attrs[inputs_name[2]]
    else:
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs)
        layer_attrs["alpha"] = inputs_name[2]
        current_inputs.append(inputs_name[2])

    graph.add_layer(
        "prim.add_", inputs=layer_inputs, outputs=layer_outputs, **layer_attrs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%i.12
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%288
    mapper._check_input(
        graph, inputs_node[1], inputs_name[1], current_outputs, add_dim=True)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.and", inputs=layer_inputs, outputs=layer_outputs)
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
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    layer_outputs = [inputs_name[0]]
    # 获取当前节点输出的list
    current_outputs = [inputs_name[0]]
    # 处理输入0，即_output_size.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["list"] = inputs_name[0]
    # 处理输入1，即v.1
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["element"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.append", inputs=layer_inputs, outputs=layer_outputs)
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
    if "pool" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["pool"] += 1
    else:
        mapper.dygraph_name_id["pool"] = 0
    pool_name = "pool" + str(mapper.dygraph_name_id["pool"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [pool_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.34
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%538
    layer_attrs["pool_size"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%539
    layer_attrs["pool_stride"] = mapper.attrs[inputs_name[2]]
    # 处理输入3，即%540
    layer_attrs["pool_padding"] = mapper.attrs[inputs_name[3]]
    # 处理输入4，即%273
    layer_attrs["ceil_mode"] = mapper.attrs[inputs_name[4]]
    # 处理输入5，即%272
    layer_attrs["exclusive"] = not mapper.attrs[inputs_name[5]]
    # 处理输入6，即%271
    graph.add_layer(
        "prim.assert",
        inputs={},
        outputs=[inputs_name[6]],
        type="eq",
        key=mapper.attrs[inputs_name[6]],
        value=None)
    layer_attrs["pool_type"] = string("avg")

    graph.add_layer(
        "fluid.dygraph.Pool2D",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    if "batchnorm" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["batchnorm"] += 1
    else:
        mapper.dygraph_name_id["batchnorm"] = 0
    batchnorm_name = "batchnorm" + str(mapper.dygraph_name_id["batchnorm"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [batchnorm_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    layer_attrs["is_test"] = True
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.80
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%778
    weights = mapper.pytorch_params[inputs_name[1]]
    mapper.paddle_params[batchnorm_name + ".weight"] = weights
    layer_attrs['num_channels'] = weights.shape[0]
    # 处理输入2，即%779
    if inputs_name[2] in mapper.pytorch_params:
        bias = mapper.pytorch_params[inputs_name[2]]
        if bias is not None:
            mapper.paddle_params[batchnorm_name + ".bias"] = bias
    else:
        mapper.paddle_params[batchnorm_name + ".bias"] = False
    # 处理输入3，即%776
    mean = mapper.pytorch_params[inputs_name[3]]
    mapper.paddle_params[batchnorm_name + "._mean"] = mean
    # 处理输入4，即%777
    var = mapper.pytorch_params[inputs_name[4]]
    mapper.paddle_params[batchnorm_name + "._variance"] = var
    # 处理输入6，即%exponential_average_factor.23
    layer_attrs["momentum"] = mapper.attrs[inputs_name[6]]
    # 处理输入7，即%766
    layer_attrs["epsilon"] = mapper.attrs[inputs_name[7]]

    graph.add_layer(
        "fluid.dygraph.BatchNorm",
        inputs=layer_inputs,
        outputs=layer_outputs,
        **layer_attrs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%13
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%12
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_attrs["axis"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    graph.add_layer(
        "fluid.layers.concat",
        inputs=layer_inputs,
        outputs=layer_outputs,
        **layer_attrs)
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
        %30 (int): 膨胀系数大小。
        %26 (int): 卷积的组数。
    """
    if "conv" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["conv"] += 1
    else:
        mapper.dygraph_name_id["conv"] = 0
    conv2d_name = "conv" + str(mapper.dygraph_name_id["conv"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [conv2d_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.8
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%25
    weights = mapper.pytorch_params[inputs_name[1]]
    mapper.paddle_params[conv2d_name + ".weight"] = weights
    layer_attrs["num_filters"] = weights.shape[0]
    layer_attrs["filter_size"] = weights.shape[2:]
    # 处理输入2，即%27
    if inputs_name[2] in mapper.pytorch_params:
        bias = mapper.pytorch_params[inputs_name[2]]
        if bias is not None:
            mapper.paddle_params[conv2d_name + ".bias"] = bias
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
    layer_attrs['num_channels'] = weights.shape[1] * mapper.attrs[inputs_name[
        6]]

    graph.add_layer(
        "fluid.dygraph.Conv2D",
        inputs=layer_inputs,
        outputs=layer_outputs,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_dim(mapper, graph, node):
    """ 构造获取维度的PaddleLayer。

    TorchScript示例:
        %106 : int = aten::dim(%101)
        参数含义:
        %106 (int): 输出，Tensor的维度。
        %101 (Tensor): 输入的Tensor。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.8
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.shape", inputs=layer_inputs, outputs=layer_outputs)
    graph.add_layer(
        "prim.len", inputs={"input": output_name}, outputs=layer_outputs)
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
    if "dropout" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["dropout"] += 1
    else:
        mapper.dygraph_name_id["dropout"] = 0
    dropout_name = "dropout" + str(mapper.dygraph_name_id["dropout"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [dropout_name, output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%119
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "fluid.dygraph.Dropout",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    if "dropout" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["dropout"] += 1
    else:
        mapper.dygraph_name_id["dropout"] = 0
    dropout_name = "dropout" + str(mapper.dygraph_name_id["dropout"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [dropout_name, output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%119
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "fluid.dygraph.Dropout",
        inputs=layer_inputs,
        outputs=layer_outputs,
        p=0.0)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%124
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%123
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.eq", inputs=layer_inputs, outputs=layer_outputs)
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

    注意：目前flatten只支持第一维的flatten
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入1，即%4
    graph.add_layer(
        "prim.assert",
        inputs={},
        outputs=[inputs_name[1]],
        type='eq',
        key=mapper.attrs[inputs_name[1]],
        value=1)
    # 处理输入2，即%2
    graph.add_layer(
        "prim.assert",
        inputs={},
        outputs=[inputs_name[2]],
        type='eq',
        key=mapper.attrs[inputs_name[2]],
        value=-1)
    # 处理输入0，即%x
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "fluid.layers.flatten",
        inputs=layer_inputs,
        outputs=layer_outputs,
        axis=1)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%72
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["list"] = inputs_name[0]
    # 处理输入1，即%88
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["index"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.getitem", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%82
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%78
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.gt", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入1，即%67
    graph.add_layer(
        "prim.assert",
        inputs={},
        outputs=[inputs_name[1]],
        type='eq',
        key=mapper.attrs[inputs_name[1]],
        value=0.0)
    # 处理输入2，即%66
    graph.add_layer(
        "prim.assert",
        inputs={},
        outputs=[inputs_name[2]],
        type='eq',
        key=mapper.attrs[inputs_name[2]],
        value=6.0)
    # 处理输入0，即%input.20
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        'fluid.layers.relu6',
        inputs=layer_inputs,
        outputs=layer_outputs,
        threshold=6.0)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%78
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%79
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.le", inputs=layer_inputs, outputs=layer_outputs)
    return current_inputs, current_outputs


def aten_len(mapper, graph, node):
    """ 构造获取list长度的PaddleLayer。

    TorchScript示例:
        %85 : int = aten::len(%83)
        参数含义:
        %85 (int): 输出，list的长度。
        %72 (list): 需要获取长度的list。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%72
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.len", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%78
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%79
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.lt", inputs=layer_inputs, outputs=layer_outputs)
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
    if "pool" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["pool"] += 1
    else:
        mapper.dygraph_name_id["pool"] = 0
    pool_name = "pool" + str(mapper.dygraph_name_id["pool"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [pool_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.11
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%20
    layer_attrs["pool_size"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%23
    layer_attrs["pool_stride"] = mapper.attrs[inputs_name[2]]
    # 处理输入3，即%21
    layer_attrs["pool_padding"] = mapper.attrs[inputs_name[3]]
    # 处理输入4，即%22
    graph.add_layer(
        "prim.assert",
        inputs={},
        outputs=[inputs_name[4]],
        type="eq",
        key=mapper.attrs[inputs_name[4]],
        value=[1, [1, 1]])
    # 处理输入5，即%19
    layer_attrs["ceil_mode"] = mapper.attrs[inputs_name[5]]
    layer_attrs["pool_type"] = string("max")

    graph.add_layer(
        "fluid.dygraph.Pool2D",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%101
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%102
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "fluid.layers.matmul", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%size_prods.38
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%114
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    current_outputs = layer_outputs

    graph.add_layer("prim.mul", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%124
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%123
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.ne", inputs=layer_inputs, outputs=layer_outputs)
    return current_inputs, current_outputs


def aten_neg(mapper, graph, node):
    """ 构造对数值取负的PaddleLayer。

    TorchScript示例:
        %909 : int = aten::neg(%908)
        参数含义:
        %909 (int): 取负后结果。
        %908 (int): 需取负的输入。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%124
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.neg", inputs=layer_inputs, outputs=layer_outputs)
    return current_inputs, current_outputs


def aten___not__(mapper, graph, node):
    """ 构造对bool型取负的PaddleLayer。

    TorchScript示例:
        %4498 : bool = aten::__not__(%aux_defined.2)
        参数含义:
        %4498 (bool): 取负后结果。
        %aux_defined.2 (bool): 需取负的输入。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%124
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.not", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "fluid.layers.relu", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "fluid.layers.relu", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "fluid.layers.relu6",
        inputs=layer_inputs,
        outputs=layer_outputs,
        threshold=6.0)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%4700
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%4703
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["shape"] = inputs_name[1]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "fluid.layers.reshape", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%18
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 处理输入1，即%8
    layer_attrs["dim"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%75
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs)
    layer_inputs["index"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.select",
        inputs=layer_inputs,
        outputs=current_outputs,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_size(mapper, graph, node):
    """ 构造获取shape的PaddleLayer。

    TorchScript示例:
        %73 : int[] = aten::size(%x.12)
        参数含义:
        %73 (list): 输出，shape的list。
        %x.12 (Tensor): 需要获取shape的Tensor。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.12
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.shape", inputs=layer_inputs, outputs=layer_outputs)
    return current_inputs, current_outputs


def aten_slice(mapper, graph, node):
    """ 构造切分list或Variable的PaddleLayer。

    TorchScript示例:
        %83 : int[] = aten::slice(%73, %82, %75, %77)
        参数含义:
        %83 (list/Tensor): 输出，切分后的list。
        %73 (list/Tensor): 需要切分的list。
        %82 (int): 切分的开始索引。
        %75 (int): 切分的结束索引。
        %77 (int): 切分的步长。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%73
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 处理输入1，即%82
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["start"] = inputs_name[1]
    # 处理输入2，即%75
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs)
    layer_inputs["end"] = inputs_name[2]
    # 处理输入3，即%77
    mapper._check_input(graph, inputs_node[3], inputs_name[3], current_outputs)
    layer_inputs["step"] = inputs_name[3]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.slice", inputs=layer_inputs, outputs=current_outputs)
    return current_inputs, current_outputs


def aten_sub(mapper, graph, node):
    """ 构造数值相减的PaddleLayer。

    TorchScript示例:
        %840 : int = aten::sub(%839, %836)
        参数含义:
        %840 (-): 相减结果。
        %839 (-): 输入数值 x。
        %836 (-): 输入数值 y。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%839
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%836
    mapper._check_input(
        graph, inputs_node[1], inputs_name[1], current_outputs, add_dim=True)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.sub", inputs=layer_inputs, outputs=layer_outputs)
    return current_inputs, current_outputs


def aten_t(mapper, graph, node):
    """ 构造矩阵转置的PaddleLayer。

    TorchScript示例:
        %840 : int = aten::sub(%839, %836)
        参数含义:
        %109 (Tensor): 输出，转置后的矩阵。
        %102 (Tensor): 需要转置的Tensor。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.12
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "fluid.layers.transpose",
        inputs=layer_inputs,
        outputs=layer_outputs,
        perm=[1, 0])
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%13
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%12
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axes"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_attrs["axes"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    graph.add_layer(
        "fluid.layers.unsqueeze",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%3
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%2
    if inputs_name[1] in mapper.attrs:
        layer_attrs["stacklevel"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_attrs["stacklevel"] = inputs_name[1]
        current_inputs.append(inputs_name[1])

    graph.add_layer(
        "prim.warnings",
        inputs=layer_inputs,
        outputs=layer_outputs,
        **layer_attrs)
    return current_inputs, current_outputs
