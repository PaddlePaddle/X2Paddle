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
        layer_inputs["pool_size"] = inputs_name[1]
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
        layer_inputs["beta"] = inputs_name[3]
        current_inputs.append(inputs_name[3])
    # 处理输入4，即%151
    if inputs_name[4] in mapper.attrs:
        layer_attrs["alpha"] = mapper.attrs[inputs_name[4]]
    else:
        mapper._check_input(graph, inputs_node[4], inputs_name[4],
                            current_outputs)
        layer_inputs["alpha"] = inputs_name[4]
        current_inputs.append(inputs_name[4])

    graph.add_layer(
        "paddle.addmm",
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
        layer_inputs["alpha"] = inputs_name[2]
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
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
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


def aten_arange(mapper, graph, node):
    """ 构造以步长均匀分隔给定数值区间的PaddleLayer。

    TorchScript示例:
        有三种情况，分别处理。
    """
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
                                current_outputs)
            layer_inputs["end"] = inputs_name[0]
            current_inputs.append(inputs_name[0])
        # 处理输入1，即%43，代表dtype
        layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[1]]]
    elif len(inputs_name) == 6:
        # %position_ids.1 : Tensor = aten::arange(%51, %52, %43, %45, %42, %46)
        # 输入的后三者分别代表layout、device、是否使用梯度
        # 处理输入0，即%51，代表start
        if inputs_name[0] in mapper.attrs:
            layer_attrs["start"] = mapper.attrs[inputs_name[0]]
        else:
            mapper._check_input(graph, inputs_node[0], inputs_name[0],
                                current_outputs)
            layer_inputs["start"] = inputs_name[0]
            current_inputs.append(inputs_name[0])
        # 处理输入1，即%52，代表end
        if inputs_name[1] in mapper.attrs:
            layer_attrs["end"] = mapper.attrs[inputs_name[1]]
        else:
            mapper._check_input(graph, inputs_node[1], inputs_name[1],
                                current_outputs)
            layer_inputs["end"] = inputs_name[1]
            current_inputs.append(inputs_name[1])
        # 处理输入2，即%43，代表dtype
        layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[2]]]
    elif len(inputs_name) == 7:
        # %position_ids.1 : Tensor = aten::arange(%51, %52, %53, %43, %45, %42, %46)
        # 输入的后三者分别代表layout、device、是否使用梯度
        # 处理输入0，即%51，代表start
        if inputs_name[0] in mapper.attrs:
            layer_attrs["start"] = mapper.attrs[inputs_name[0]]
        else:
            mapper._check_input(graph, inputs_node[0], inputs_name[0],
                                current_outputs)
            layer_inputs["start"] = inputs_name[0]
            current_inputs.append(inputs_name[0])
        # 处理输入1，即%52，代表end
        if inputs_name[1] in mapper.attrs:
            layer_attrs["end"] = mapper.attrs[inputs_name[1]]
        else:
            mapper._check_input(graph, inputs_node[1], inputs_name[1],
                                current_outputs)
            layer_inputs["end"] = inputs_name[1]
            current_inputs.append(inputs_name[1])
        # 处理输入2，即%53，代表step
        if inputs_name[2] in mapper.attrs:
            layer_attrs["step"] = mapper.attrs[inputs_name[2]]
        else:
            mapper._check_input(graph, inputs_node[2], inputs_name[2],
                                current_outputs)
            layer_inputs["step"] = inputs_name[2]
            current_inputs.append(inputs_name[2])
        # 处理输入3，即%43，代表dtype
        layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[3]]]
    else:
        raise Exception("Unknown aten::arange signature taking " + str(
            len(inputs_name)) + " arguments.")

    graph.add_layer(
        "paddle.arange",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
        "paddle.nn.Pool2D",
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
        "paddle.nn.BatchNorm",
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
        layer_inputs["axis"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    graph.add_layer(
        "fluid.layers.concat",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.170
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%720
    if inputs_name[1] in mapper.attrs:
        layer_attrs["num_or_sections"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_inputs["num_or_sections"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入2，即%719
    if inputs_name[2] in mapper.attrs:
        layer_attrs["dim"] = mapper.attrs[inputs_name[2]]
    else:
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs)
        layer_inputs["dim"] = inputs_name[2]
        current_inputs.append(inputs_name[2])
    graph.add_layer(
        "fluid.layers.split",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%50
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 处理输入1，即%name.1
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["element"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.contain", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%4058
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.equal", inputs=layer_inputs, outputs=layer_outputs)
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
        "paddle.nn.Conv2D",
        inputs=layer_inputs,
        outputs=layer_outputs,
        **layer_attrs)
    return current_inputs, current_outputs


def aten__convolution(mapper, graph, node):
    """ 构造conv2d的PaddleLayer。

    TorchScript示例:
        %input.10 : Tensor = aten::_convolution(%input.8, %25, %27, %28, %29, %30, %26)
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
        "paddle.nn.Conv2D",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%sinusoid_inp.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("paddle.cos", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%end.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    graph.add_layer("prim.equal", inputs=layer_inputs, outputs=layer_outputs)

    return current_inputs, current_outputs


def aten_dict(mapper, graph, node):
    """ 构造初始化dict的PaddleLayer。

    TorchScript示例:
        %features.1 : Dict(str, Tensor) = aten::dict()
        参数含义:
        %features.1: 输出，初始化的dict。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    current_inputs = {}
    # 获取当前节点输出的list
    current_outputs = [output_name]

    graph.add_layer("prim.dict", inputs=layer_inputs, outputs=layer_outputs)
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
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%123
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.div", inputs=layer_inputs, outputs=layer_outputs)
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

    graph.add_layer("prim.div", inputs=layer_inputs, outputs=layer_outputs)
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
        "paddle.nn.Dropout", inputs=layer_inputs, outputs=layer_outputs, p=0.0)
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
        "paddle.nn.Dropout", inputs=layer_inputs, outputs=layer_outputs, p=0.0)
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
    if "embedding" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["embedding"] += 1
    else:
        mapper.dygraph_name_id["embedding"] = 0
    embedding_name = "embedding" + str(mapper.dygraph_name_id["embedding"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [embedding_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%57
    weights = mapper.pytorch_params[inputs_name[0]]
    mapper.paddle_params[embedding_name + ".weight"] = weights
    layer_attrs["size"] = weights.shape
    # 处理输入1，即%input_ids.1
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["input"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入2，即%45
    layer_attrs["padding_idx"] = mapper.attrs[inputs_name[2]]
    # 处理输入4，即%46
    layer_attrs["is_sparse"] = mapper.attrs[inputs_name[4]]

    graph.add_layer(
        "paddle.nn.Embedding",
        inputs=layer_inputs,
        outputs=layer_outputs,
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


def aten_exp(mapper, graph, node):
    """ 构造以自然数e为底指数运算的PaddleLayer。

    TorchScript示例:
        %55 : Tensor = aten::tanh(%54)
        参数含义:
        %55 (Tensor): 输出，运算后的结果。
        %54 (Tensor): 需要指数运算的Tensor。
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
        "fluid.layers.exp", inputs=layer_inputs, outputs=layer_outputs)
    return current_inputs, current_outputs


def aten_expand(mapper, graph, node):
    """ 构造复制维度的PaddleLayer。

    TorchScript示例:
        %1889 : Tensor = aten::expand(%1875, %1888, %1567)
        参数含义:
        %1889 (Tensor): 复制后的结果。
        %1875 (Tensor): 需要复制的Tensor。
        %1567 (bool): 未使用。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%1875
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%1888
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    graph.add_layer(
        "fluid.layers.create_global_var",
        inputs={"shape": inputs_name[1]},
        outputs=[inputs_name[1] + "_var"],
        value=1.0,
        dtype=string("int64"),
        persistable=True)
    layer_inputs["target_tensor"] = inputs_name[1] + "_var"
    current_outputs.append(inputs_name[1] + "_var")
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    current_inputs.append(inputs_name[1])

    graph.add_layer(
        "fluid.layers.expand_as", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%49
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["num_rows"] = inputs_name[0]
    if len(inputs_name) > 5:
        # 处理输入1，即%_50
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_inputs["num_columns"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理倒数第4个输入，即%15
    layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[-4]]]

    graph.add_layer(
        "fluid.layers.eye",
        inputs=layer_inputs,
        outputs=layer_outputs,
        **layer_attrs)
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


def aten_Float(mapper, graph, node):
    """ 构造取浮点型的PaddleLayer。

    TorchScript示例:
        %3992 : float = aten::Float(%3991)
        参数含义:
        %3992 (int): 向上取整后的整数。
        %3991 (float): 需要取整的浮点数。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%3991
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.float", inputs=layer_inputs, outputs=layer_outputs)
    return current_inputs, current_outputs


def aten_floor(mapper, graph, node):
    """ 构造向上取整的PaddleLayer。

    TorchScript示例:
        %3978 : int = aten::floor(%scale.18)
        参数含义:
        %3978 (int): 向上取整后的整数。
        %scale.18 (float): 需要取整的浮点数。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%scale.18
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.floor", inputs=layer_inputs, outputs=layer_outputs)
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

    graph.add_layer("prim.floordiv", inputs=layer_inputs, outputs=layer_outputs)
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

    graph.add_layer("prim.floordiv", inputs=layer_inputs, outputs=layer_outputs)
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
    if "gelu" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["gelu"] += 1
    else:
        mapper.dygraph_name_id["gelu"] = 0
    gelu_name = "gelu" + str(mapper.dygraph_name_id["gelu"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [gelu_name, output_name]
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
        "paddle.nn.GELU", inputs=layer_inputs, outputs=layer_outputs)
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
    if "tanh" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["tanh"] += 1
    else:
        mapper.dygraph_name_id["tanh"] = 0
    tanh_name = "tanh" + str(mapper.dygraph_name_id["tanh"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [tanh_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.20
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%67
    layer_attrs["min"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%66
    layer_attrs["max"] = mapper.attrs[inputs_name[2]]

    graph.add_layer(
        'paddle.nn.Hardtanh',
        inputs=layer_inputs,
        outputs=layer_outputs,
        **layer_attrs)
    return current_inputs, current_outputs


def aten_index_select(mapper, graph, node):
    """ 构造对dict加入元素的PaddleLayer。

    TorchScript示例:
        %bd.3 : Tensor = aten::index_select(%x2.3, %320, %371)
        参数含义:
        %bd.3 (Tensor): 输出，选择后的Tensor。
        %x2.3 (Tensor): 需要选择的Tensor。
        %320 (int): 维度。
        %371 (Tensor): 选择的索引。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x2.3
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%320
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_inputs["axis"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入2，即%371
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs)
    layer_inputs["index"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "prim.index_select",
        inputs=layer_inputs,
        outputs=current_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%1738
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.int", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%size.122
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%3931
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.is", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%size.122
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%3931
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["y"] = inputs_name[1]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.isnot", inputs=layer_inputs, outputs=layer_outputs)
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
    if "layernorm" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["layernorm"] += 1
    else:
        mapper.dygraph_name_id["layernorm"] = 0
    layernorm_name = "layernorm" + str(mapper.dygraph_name_id["layernorm"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [layernorm_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%input.6
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%1181
    layer_attrs["normalized_shape"] = mapper.attrs[inputs_name[1]]
    # 处理输入2，即%174
    weights = mapper.pytorch_params[inputs_name[2]]
    mapper.paddle_params[layernorm_name + ".weight"] = weights
    # 处理输入3，即%173
    if inputs_name[3] in mapper.pytorch_params:
        bias = mapper.pytorch_params[inputs_name[3]]
        if bias is not None:
            mapper.paddle_params[layernorm_name + ".bias"] = bias
    else:
        mapper.paddle_params[layernorm_name + ".bias"] = False
    # 处理输入4，即%70
    layer_attrs["epsilon"] = mapper.attrs[inputs_name[4]]

    graph.add_layer(
        "paddle.nn.LayerNorm",
        inputs=layer_inputs,
        outputs=layer_outputs,
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


def aten_leaky_relu_(mapper, graph, node):
    """ 构造leaky relu激活的PaddleLayer。

    TorchScript示例:
        %input.117 : Tensor = aten::leaky_relu_(%input.114, %1570)
        参数含义:
        %input.117 (Tensor): 输出，leaky relu后的结果。
        %input.114 (Tensor): 需要leaky relu的Tensor。
        %1570 (float): 输入中的元素小于0时的斜率。
    """
    if "leaky_relu" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["leaky_relu"] += 1
    else:
        mapper.dygraph_name_id["leaky_relu"] = 0
    leaky_relu_name = "leaky_relu" + str(mapper.dygraph_name_id["leaky_relu"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [leaky_relu_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.5
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%1570
    layer_attrs["negative_slope"] = mapper.attrs[inputs_name[1]]

    graph.add_layer(
        "paddle.nn.LeakyReLU",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
        "paddle.nn.Pool2D",
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

    graph.add_layer("paddle.matmul", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%result.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%4967
    if inputs_name[1] in mapper.attrs:
        layer_attrs["dim"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_inputs["dim"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入2，即%3
    if inputs_name[1] in mapper.attrs:
        layer_attrs["keep_dim"] = mapper.attrs[inputs_name[2]]
    else:
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs)
        layer_inputs["keep_dim"] = inputs_name[2]
        current_inputs.append(inputs_name[2])

    graph.add_layer(
        "fluid.layers.reduce_mean",
        inputs=layer_inputs,
        outputs=layer_outputs,
        **layer_attrs)
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


def aten_permute(mapper, graph, node):
    """ 构造对bool型取负的PaddleLayer。

    TorchScript示例:
        %2385 : Tensor = aten::permute(%cls_confs0.2, %2384)
        参数含义:
        %2385 (Tensor): 重排后的结果。
        %cls_confs0.2 (Tensor): 需要重排的Tensor。
        %2348 (list): 依照此参数进行重排。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%cls_confs0.2
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%2348
    if inputs_name[1] in mapper.attrs:
        layer_attrs["perm"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_inputs["perm"] = inputs_name[1]
        current_inputs.append(inputs_name[1])

    graph.add_layer(
        "fluid.layers.transpose",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%4703
    if inputs_name[1] in mapper.attrs:
        layer_attrs["factor"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_inputs["factor"] = inputs_name[1]
        current_inputs.append(inputs_name[1])

    graph.add_layer(
        "fluid.layers.pow",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    if "relu" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["relu"] += 1
    else:
        mapper.dygraph_name_id["relu"] = 0
    relu_name = "relu" + str(mapper.dygraph_name_id["relu"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [relu_name, output_name]
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
        "paddle.nn.ReLU", inputs=layer_inputs, outputs=layer_outputs)
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
    if "relu" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["relu"] += 1
    else:
        mapper.dygraph_name_id["relu"] = 0
    relu_name = "relu" + str(mapper.dygraph_name_id["relu"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [relu_name, output_name]
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
        "paddle.nn.ReLU", inputs=layer_inputs, outputs=layer_outputs)
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
    if "relu6" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["relu6"] += 1
    else:
        mapper.dygraph_name_id["relu6"] = 0
    relu6_name = "relu6" + str(mapper.dygraph_name_id["relu6"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [relu6_name, output_name]
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
        "paddle.nn.ReLU6", inputs=layer_inputs, outputs=layer_outputs)
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
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%4703
    if inputs_name[1] in mapper.attrs:
        layer_attrs["shape"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_inputs["shape"] = inputs_name[1]
        current_inputs.append(inputs_name[1])

    graph.add_layer(
        "fluid.layers.reshape",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%30
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%13
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["y"] = inputs_name[1]
    # 处理输入2，即%7
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs)
    layer_inputs["alpha"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.rsub", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%end.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    input_type = list(node.inputs())[0].type()
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    if str(input_type) == "Tensor":
        graph.add_layer(
            "prim.equal", inputs=layer_inputs, outputs=layer_outputs)
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


def aten__set_item(mapper, graph, node):
    """ 构造对dict加入元素的PaddleLayer。

    TorchScript示例:
        = aten::_set_item(%features.1, %out_name.1, %x.3)
        参数含义:
        %features.1 (list): dict。
        %out_name.1 (-): dict的key。
        %x.3 (-): dict的value。
    """
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = []
    # 处理输入0，即%features.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["dict"] = inputs_name[0]
    # 处理输入1，即%out_name.1
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["key"] = inputs_name[1]
    # 处理输入2，即%x.3
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs)
    layer_inputs["value"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("prim.set_item", inputs=layer_inputs, outputs=[])
    return current_inputs, current_outputs


def aten_sigmoid(mapper, graph, node):
    """ 构造sigmoid激活的PaddleLayer。

    TorchScript示例:
        %55 : Tensor = aten::sigmoid(%54)
        参数含义:
        %55 (Tensor): 输出，sigmoid后的结果。
        %54 (Tensor): 需要tanh的Tensor。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%54
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "fluid.layers.sigmoid", inputs=layer_inputs, outputs=layer_outputs)
    return current_inputs, current_outputs


def aten_sin(mapper, graph, node):
    """ 构造数学计算sin的PaddleLayer。

    TorchScript示例:
        %94 : Tensor = aten::sin(%sinusoid_inp.1)
        参数含义:
        %94 (Tensor): 输出，sin之后的结果。
        %sinusoid_inp.1 (Tensor): 需要进行shape的Tensor。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%sinusoid_inp.1
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("paddle.sin", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.12
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    if len(inputs_name) > 1:
        # 处理输入1，即%12
        if inputs_name[1] in mapper.attrs:
            layer_attrs["dim"] = mapper.attrs[inputs_name[1]]
        else:
            mapper._check_input(graph, inputs_node[1], inputs_name[1],
                                current_outputs)
            layer_inputs["dim"] = inputs_name[1]
            current_inputs.append(inputs_name[1])
        graph.add_layer(
            "prim.shape_dim",
            inputs=layer_inputs,
            outputs=layer_outputs,
            **layer_attrs)
        return current_inputs, current_outputs

    graph.add_layer("prim.shape", inputs=layer_inputs, outputs=layer_outputs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    if len(inputs_name) == 5:
        # 处理输入0，即%73
        mapper._check_input(graph, inputs_node[0], inputs_name[0],
                            current_outputs)
        layer_inputs["input"] = inputs_name[0]

        # 获取当前节点输入的list
        current_inputs = list(layer_inputs.values())
        # 处理输入1，即%_81
        if inputs_name[1] in mapper.attrs:
            graph.add_layer(
                "prim.list",
                inputs={},
                outputs=[inputs_name[1] + "_list"],
                input0=mapper.attrs[inputs_name[1]])
        else:
            mapper._check_input(graph, inputs_node[1], inputs_name[1],
                                current_outputs)
            graph.add_layer(
                "prim.list",
                inputs={"input0": inputs_name[1]},
                outputs=[inputs_name[1] + "_list"])
            current_inputs.append(inputs_name[1])
        layer_inputs["axes"] = inputs_name[1] + "_list"
        current_inputs.append(inputs_name[1] + "_list")
        current_outputs.append(inputs_name[1] + "_list")
        # 处理输入3，即%82
        if inputs_name[2] in mapper.attrs:
            graph.add_layer(
                "prim.list",
                inputs={},
                outputs=[inputs_name[2] + "_list"],
                input0=mapper.attrs[inputs_name[2]])
        else:
            mapper._check_input(graph, inputs_node[2], inputs_name[2],
                                current_outputs)
            graph.add_layer(
                "prim.list",
                inputs={"input0": inputs_name[2]},
                outputs=[inputs_name[2] + "_list"])
            current_inputs.append(inputs_name[2])
        layer_inputs["starts"] = inputs_name[2] + "_list"
        current_inputs.append(inputs_name[2] + "_list")
        current_outputs.append(inputs_name[2] + "_list")
        # 处理输入3，即%85
        if inputs_name[3] in mapper.attrs:
            if 9223372036854775807 == mapper.attrs[inputs_name[3]]:
                import math
                input0 = int(math.pow(2, 31) - 1)
            graph.add_layer(
                "prim.list",
                inputs={},
                outputs=[inputs_name[3] + "_list"],
                input0=input0)
        else:
            mapper._check_input(graph, inputs_node[3], inputs_name[3],
                                current_outputs)
            graph.add_layer(
                "prim.list",
                inputs={"input0": inputs_name[3]},
                outputs=[inputs_name[3] + "_list"])
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
                input0=mapper.attrs[inputs_name[4]])
        else:
            mapper._check_input(graph, inputs_node[4], inputs_name[4],
                                current_outputs)
            graph.add_layer(
                "prim.list",
                inputs={"input0": inputs_name[4]},
                outputs=[inputs_name[4] + "_list"])
            current_inputs.append(inputs_name[4])
        layer_inputs["strides"] = inputs_name[4] + "_list"
        current_inputs.append(inputs_name[4] + "_list")
        current_outputs.append(inputs_name[4] + "_list")

        graph.add_layer(
            "fluid.layers.strided_slice",
            inputs=layer_inputs,
            outputs=layer_outputs)
    else:
        # 处理输入0，即%73
        mapper._check_input(graph, inputs_node[0], inputs_name[0],
                            current_outputs)
        layer_inputs["input"] = inputs_name[0]
        # 处理输入1，即%82
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_inputs["start"] = inputs_name[1]
        # 处理输入2，即%75
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs)
        layer_inputs["end"] = inputs_name[2]
        # 处理输入3，即%77
        mapper._check_input(graph, inputs_node[3], inputs_name[3],
                            current_outputs)
        layer_inputs["step"] = inputs_name[3]
        # 获取当前节点输入的list
        current_inputs = list(layer_inputs.values())

        graph.add_layer(
            "prim.slice", inputs=layer_inputs, outputs=layer_outputs)
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
    if "softmax" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["softmax"] += 1
    else:
        mapper.dygraph_name_id["softmax"] = 0
    softmax_name = "softmax" + str(mapper.dygraph_name_id["softmax"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [softmax_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.31
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    layer_attrs["axis"] = mapper.attrs[inputs_name[1]]

    graph.add_layer(
        "paddle.nn.Softmax",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    if "softplus" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["softplus"] += 1
    else:
        mapper.dygraph_name_id["softplus"] = 0
    softplus_name = "softplus" + str(mapper.dygraph_name_id["softplus"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [softplus_name, output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.31
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    layer_attrs["beta"] = mapper.attrs[inputs_name[1]]
    layer_attrs["threshold"] = mapper.attrs[inputs_name[2]]

    graph.add_layer(
        "paddle.nn.Softplus",
        inputs=layer_inputs,
        outputs=layer_outputs,
        **layer_attrs)
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


def aten_tanh(mapper, graph, node):
    """ 构造tanh激活的PaddleLayer。

    TorchScript示例:
        %55 : Tensor = aten::tanh(%54)
        参数含义:
        %55 (Tensor): 输出，tanh后的结果。
        %54 (Tensor): 需要tanh的Tensor。
    """
    if "tanh" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["tanh"] += 1
    else:
        mapper.dygraph_name_id["tanh"] = 0
    tanh_name = "tanh" + str(mapper.dygraph_name_id["tanh"])
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [tanh_name, output_name]
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
        "paddle.nn.Tanh", inputs=layer_inputs, outputs=layer_outputs)
    return current_inputs, current_outputs


def aten_split(mapper, graph, node):
    """ 构造分割Tensor的PaddleLayer。

    TorchScript示例:
        %160 : Tensor[] = aten::split(%159, %135, %123)
        参数含义:
        %160 (Tensor): 输出，分割后的矩阵。
        %159 (Tensor): 需要分割的Tensor。
        %135 (int): 分割的数量。
        %723 (int): 轴。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%159
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 处理输入2，即%723
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs)
    layer_inputs["dim"] = inputs_name[2]
    # 处理输入1，即%135
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    input_type = list(node.inputs())[0].type()
    if "[]" in str(input_type):
        layer_inputs["num_or_sections"] = inputs_name[1]
    else:
        graph.add_layer(
            "prim.shape",
            inputs={"input": inputs_name[0]},
            outputs=[inputs_name[1] + "_shape"])
        graph.add_layer(
            "prim.getitem",
            inputs={
                "list": inputs_name[1] + "_shape",
                "index": inputs_name[2]
            },
            outputs=[inputs_name[1] + "_item"])
        graph.add_layer(
            "prim.div",
            inputs={"x": inputs_name[1] + "_item",
                    "y": inputs_name[1]},
            outputs=[inputs_name[1] + "_div"])
        graph.add_layer(
            "prim.int",
            inputs={"input": inputs_name[1] + "_div"},
            outputs=[inputs_name[1] + "_int"])
        layer_inputs["num_or_sections"] = inputs_name[1] + "_int"
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer(
        "fluid.layers.split",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%13
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%12
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_inputs["axis"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    graph.add_layer(
        "paddle.stack",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.21
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 处理输入1，即%704
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    dim1 = inputs_name[1]
    # 处理输入2，即%705
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs)
    dim2 = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    graph.add_layer(
        "prim.shape",
        inputs={"input": inputs_name[0]},
        outputs=[output_name + "_shape"])
    current_outputs.append(output_name + "_shape")
    graph.add_layer(
        "prim.len",
        inputs={"input": output_name + "_shape"},
        outputs=[output_name + "_len"])
    current_outputs.append(output_name + "_len")
    current_inputs.append(output_name + "_shape")
    graph.add_layer(
        "prim.len2list",
        inputs={"len": output_name + "_len"},
        outputs=[output_name + "_list"])
    current_outputs.append(output_name + "_list")
    current_inputs.append(output_name + "_len")
    graph.add_layer(
        "prim.check_dim",
        inputs={"len": output_name + "_len",
                "dim": dim1},
        outputs=[dim1 + "_new"])
    graph.add_layer(
        "prim.check_dim",
        inputs={"len": output_name + "_len",
                "dim": dim2},
        outputs=[dim2 + "_new"])
    graph.add_layer(
        "prim.replaceitem",
        inputs={
            "list": output_name + "_list",
            "index": dim1 + "_new",
            "item": dim2 + "_new"
        },
        outputs=[])
    graph.add_layer(
        "prim.replaceitem",
        inputs={
            "list": output_name + "_list",
            "index": dim2 + "_new",
            "item": dim1 + "_new"
        },
        outputs=[])
    graph.add_layer(
        "fluid.layers.transpose",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    assert len(inputs_name) == 5, "Paddle only support converting the dtype!"
    # 处理输入0，即%13
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%12
    layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[1]]]

    graph.add_layer(
        "fluid.layers.cast",
        inputs=layer_inputs,
        outputs=layer_outputs,
        **layer_attrs)
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
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%12
    if inputs_name[1] in mapper.attrs:
        layer_attrs["axis"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_inputs["axis"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    graph.add_layer(
        "paddle.tensor.unsqueeze",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
        %4995 (float): 宽度的乘数因子。
    """
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.13
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["input"] = inputs_name[0]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%4963
    if inputs_name[1] in mapper.attrs:
        layer_attrs["out_shape"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_inputs["out_shape"] = inputs_name[1]
        current_inputs.append(inputs_name[1])
    # 处理输入2，即%5421
    if inputs_name[2] in mapper.attrs:
        layer_attrs["align_corners"] = mapper.attrs[inputs_name[2]]
    else:
        mapper._check_input(graph, inputs_node[2], inputs_name[2],
                            current_outputs)
        layer_inputs["align_corners"] = inputs_name[2]
        current_inputs.append(inputs_name[2])
    # 处理输入3和4，构造assert
    list_layer_inputs = {}
    mapper._check_input(graph, inputs_node[3], inputs_name[3], current_outputs)
    list_layer_inputs["key"] = inputs_name[3]
    current_inputs.append(inputs_name[3])
    mapper._check_input(graph, inputs_node[4], inputs_name[4], current_outputs)
    list_layer_inputs["value"] = inputs_name[4]
    current_inputs.append(inputs_name[4])
    graph.add_layer(
        "prim.assert",
        inputs=list_layer_inputs,
        outputs=[output_name + "_assert"],
        type="eq")
    layer_inputs["scale"] = inputs_name[3]
    layer_attrs["align_mode"] = 0
    graph.add_layer(
        "fluid.layers.interpolate",
        inputs=layer_inputs,
        outputs=layer_outputs,
        **layer_attrs)
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    layer_attrs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%x.20
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["x"] = inputs_name[0]
    # 获取当前节点输入、输出的list
    current_inputs = list(layer_inputs.values())
    # 处理输入1，即%430
    if inputs_name[1] in mapper.attrs:
        layer_attrs["shape"] = mapper.attrs[inputs_name[1]]
    else:
        mapper._check_input(graph, inputs_node[1], inputs_name[1],
                            current_outputs)
        layer_inputs["shape"] = inputs_name[1]
        current_inputs.append(inputs_name[1])

    graph.add_layer(
        "fluid.layers.reshape",
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
        layer_inputs["stacklevel"] = inputs_name[1]
        current_inputs.append(inputs_name[1])

    graph.add_layer(
        "prim.warnings",
        inputs=layer_inputs,
        outputs=layer_outputs,
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
    output_name = mapper._get_outputs_name(node)[0]
    layer_outputs = [output_name]
    layer_inputs = {}
    inputs_name, inputs_node = mapper._get_inputs_name(node)
    # 获取当前节点输出的list
    current_outputs = [output_name]
    # 处理输入0，即%209
    mapper._check_input(graph, inputs_node[0], inputs_name[0], current_outputs)
    layer_inputs["condition"] = inputs_name[0]
    # 处理输入1，即%w0.2
    mapper._check_input(graph, inputs_node[1], inputs_name[1], current_outputs)
    layer_inputs["x"] = inputs_name[1]
    # 处理输入1，即%w0.2
    mapper._check_input(graph, inputs_node[2], inputs_name[2], current_outputs)
    layer_inputs["y"] = inputs_name[2]
    # 获取当前节点输入的list
    current_inputs = list(layer_inputs.values())

    graph.add_layer("paddle.where", inputs=layer_inputs, outputs=layer_outputs)
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
                            current_outputs)
        layer_inputs["shape"] = inputs_name[0]
        current_inputs.append(inputs_name[0])
    # 处理输入1，即%8，代表dtype
    layer_attrs["dtype"] = dtype_dict[mapper.attrs[inputs_name[1]]]

    graph.add_layer(
        "paddle.zeros",
        inputs=layer_inputs,
        outputs=layer_outputs,
        **layer_attrs)
    return current_inputs, current_outputs