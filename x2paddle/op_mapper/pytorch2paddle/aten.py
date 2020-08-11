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

    PyTorch Script 示例:
        %x.5 : Tensor = aten::adaptive_avg_pool2d(%x.3, %_output_size.1)
        参数含义:
        %x.5 (Tensor): 池化后结果Tensor。
        %x.3 (Tensor): 输入Tensor。
        %_output_size.1 (list): 自适应池化后的Tensor的宽、高大小。
    """
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    adapoo2d_inputs = []
    input_node = list(node.inputs())[0].node()
    script_input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.outputs_info[script_input_unique_id]
    mapper._check_input(graph, input_node, input_node_name, node_outputs)
    adapoo2d_inputs.append(input_node_name)
    attr_node = list(node.inputs())[1].node()
    attr_unique_id = list(node.inputs())[1].unique()
    attr_node_name = mapper.outputs_info[attr_unique_id]
    attrs = {}
    attrs["pool_size"] = mapper.attrs[
        attr_node_name] if attr_node_name in mapper.attrs else attr_node_name
    if attr_node_name not in mapper.attrs:
        adapoo2d_inputs.append(attr_node_name)
    attrs["pool_type"] = string("avg")
    graph.add_layer(
        "fluid.layers.adaptive_pool2d",
        inputs={"input": input_node_name},
        outputs=[output_name],
        **attrs)
    return [input_node_name], node_outputs


def aten_addmm(mapper, graph, node):
    """ 构造addmm的PaddleLayer，该节点实现out = alpha ∗ x ∗ y + beta ∗ input。

    PyTorch Script 示例:
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
    inputs = {}
    attrs = {}
    addmm_inputs = []
    node_outputs = [output_name]
    input_node = list(node.inputs())[0].node()
    script_input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.outputs_info[script_input_unique_id]
    mapper._check_input(
        graph, input_node, input_node_name, node_outputs, add_dim=True)
    inputs['input'] = input_node_name
    addmm_inputs.append(input_node_name)
    x_node = list(node.inputs())[1].node()
    x_unique_id = list(node.inputs())[1].unique()
    x_node_name = mapper.outputs_info[x_unique_id]
    mapper._check_input(graph, x_node, x_node_name, node_outputs)
    inputs['x'] = x_node_name
    addmm_inputs.append(x_node_name)
    y_node = list(node.inputs())[2].node()
    y_unique_id = list(node.inputs())[2].unique()
    y_node_name = mapper.outputs_info[y_unique_id]
    mapper._check_input(graph, y_node, y_node_name, node_outputs)
    inputs['y'] = y_node_name
    addmm_inputs.append(y_node_name)
    beta_node = list(node.inputs())[3].node()
    beta_unique_id = list(node.inputs())[3].unique()
    beta_node_name = mapper.outputs_info[beta_unique_id]
    attrs['beta'] = mapper.attrs[
        beta_node_name] if beta_node_name in mapper.attrs else beta_node_name
    if beta_node_name not in mapper.attrs:
        addmm_inputs.append(beta_node_name)
    alpha_node = list(node.inputs())[4].node()
    alpha_unique_id = list(node.inputs())[4].unique()
    alpha_node_name = mapper.outputs_info[alpha_unique_id]
    attrs['alpha'] = mapper.attrs[
        alpha_node_name] if alpha_node_name in mapper.attrs else alpha_node_name
    if alpha_node_name not in mapper.attrs:
        addmm_inputs.append(alpha_node_name)
    graph.add_layer(
        "fluid.layers.addmm", inputs=inputs, outputs=[output_name], **attrs)
    return addmm_inputs, node_outputs


def aten_add_(mapper, graph, node):
    """ 构造add的PaddleLayer，该节点实现out = x + alpha * y。

    PyTorch Script 示例:
        %output.5 : Tensor = aten::add_(%output.2, %150, %151)
        参数含义:
        %output.5 (Tensor): add结果Tensor。
        %output.2 (Tensor): 输入Tensor x。
        %150 (Tensor): 输入Tensor y。
        %151 (int/float): 输入alpha。
    """
    output_name = mapper._get_outputs_name(node)[0]
    inputs = {}
    attrs = {}
    add_inputs = []
    node_outputs = [output_name]
    x_node = list(node.inputs())[0].node()
    x_unique_id = list(node.inputs())[0].unique()
    x_node_name = mapper.outputs_info[x_unique_id]
    mapper._check_input(graph, x_node, x_node_name, node_outputs)
    inputs['x'] = x_node_name
    add_inputs.append(x_node_name)
    y_node = list(node.inputs())[1].node()
    y_unique_id = list(node.inputs())[1].unique()
    y_node_name = mapper.outputs_info[y_unique_id]
    mapper._check_input(graph, y_node, y_node_name, node_outputs, add_dim=True)
    inputs['y'] = y_node_name
    add_inputs.append(y_node_name)
    alpha_node = list(node.inputs())[2].node()
    alpha_unique_id = list(node.inputs())[2].unique()
    alpha_node_name = mapper.outputs_info[alpha_unique_id]
    attrs['alpha'] = mapper.attrs[
        alpha_node_name] if alpha_node_name in mapper.attrs else alpha_node_name
    if alpha_node_name not in mapper.attrs:
        add_inputs.append(alpha_node_name)
    graph.add_layer("prim.add", inputs=inputs, outputs=[output_name], **attrs)
    return add_inputs, node_outputs


def aten_append(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    inputs = {}
    for i, input_ivalue in enumerate(node.inputs()):
        input_node = input_ivalue.node()
        input_unique_id = input_ivalue.unique()
        input_node_name = mapper.outputs_info[input_unique_id]
        mapper._check_input(graph, input_node, input_node_name, node_outputs)
        if i == 0:
            inputs['list'] = input_node_name
        else:
            inputs['element'] = input_node_name
    graph.add_layer("prim.append", inputs=inputs, outputs=[output_name])
    return list(inputs.values()), node_outputs


def aten_conv2d(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    inputs = {}
    attrs = {}
    conv2d_inputs = []
    node_outputs = [output_name]
    if "conv" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["conv"] += 1
    else:
        mapper.dygraph_name_id["conv"] = 0
    conv2d_name = "conv" + str(mapper.dygraph_name_id["conv"])
    # 输入input
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.outputs_info[input_unique_id]
    inputs['input'] = input_node_name
    conv2d_inputs.append(input_node_name)
    # 输入weight
    weight_node = list(node.inputs())[1].node()
    weight_unique_id = list(node.inputs())[1].unique()
    weight_node_name = mapper.outputs_info[weight_unique_id]
    weights = mapper.pytorch_params[weight_node_name]
    mapper.paddle_params[conv2d_name + '.weight'] = weights
    attrs['num_filters'] = weights.shape[0]
    attrs['filter_size'] = weights.shape[2:]
    # 输入bias
    bias_node = list(node.inputs())[2].node()
    bias_unique_id = list(node.inputs())[2].unique()
    bias_node_name = mapper.outputs_info[bias_unique_id]
    if bias_node_name in mapper.pytorch_params:
        bias = mapper.pytorch_params[bias_node_name]
        mapper.paddle_params[conv2d_name + '.bias'] = bias
    else:
        mapper.paddle_params[conv2d_name + '.bias'] = False
    # 输入stride
    stride_node = list(node.inputs())[3].node()
    stride_unique_id = list(node.inputs())[3].unique()
    stride_node_name = mapper.outputs_info[stride_unique_id]
    attrs['stride'] = mapper.attrs[stride_node_name]
    # 输入padding
    padding_node = list(node.inputs())[4].node()
    padding_unique_id = list(node.inputs())[4].unique()
    padding_node_name = mapper.outputs_info[padding_unique_id]
    attrs['padding'] = mapper.attrs[padding_node_name]
    # 输入dilation
    dilation_node = list(node.inputs())[5].node()
    dilation_unique_id = list(node.inputs())[5].unique()
    dilation_node_name = mapper.outputs_info[dilation_unique_id]
    attrs['dilation'] = mapper.attrs[dilation_node_name]
    # 输入group
    groups_node = list(node.inputs())[6].node()
    groups_unique_id = list(node.inputs())[6].unique()
    groups_node_name = mapper.outputs_info[groups_unique_id]
    attrs['groups'] = mapper.attrs[groups_node_name]
    attrs['num_channels'] = weights.shape[1] * mapper.attrs[groups_node_name]
    graph.add_layer(
        "fluid.dygraph.Conv2D",
        inputs=inputs,
        outputs=[conv2d_name, output_name],
        **attrs)
    return conv2d_inputs, node_outputs


def aten_dim(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.outputs_info[input_unique_id]
    mapper._check_input(graph, input_node, input_node_name, node_outputs)
    graph.add_layer(
        "prim.shape", inputs={'input': input_node_name}, outputs=[output_name])
    graph.add_layer(
        "prim.len", inputs={'input': output_name}, outputs=[output_name])
    return [input_node_name], node_outputs


def aten_dropout(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    if "dropout" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["dropout"] += 1
    else:
        mapper.dygraph_name_id["dropout"] = 0
    dropout_name = "dropout" + str(mapper.dygraph_name_id["dropout"])
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.outputs_info[input_unique_id]
    mapper._check_input(graph, input_node, input_node_name, node_outputs)
    graph.add_layer(
        "fluid.dygraph.Dropout",
        inputs={"input": input_node_name},
        outputs=[dropout_name, output_name],
        p=0.0)
    return [input_node_name], node_outputs


def aten_eq(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    inputs = {}
    eq_inputs = []
    for i, input_ivalue in enumerate(node.inputs()):
        input_node = input_ivalue.node()
        input_unique_id = input_ivalue.unique()
        input_node_name = mapper.outputs_info[input_unique_id]
        mapper._check_input(graph, input_node, input_node_name, node_outputs)
        inputs['eq{}'.format(i)] = input_node_name
        eq_inputs.append(input_node_name)
    graph.add_layer("prim.eq", inputs=inputs, outputs=[output_name])
    return list(inputs.values()), node_outputs


def aten_flatten(mapper, graph, node):
    # 目前只支持第一维的flatten
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    flatten_inputs = []
    for i, input_ivalue in enumerate(node.inputs()):
        if i == 0:
            continue
        input_node = input_ivalue.node()
        input_unique_id = input_ivalue.unique()
        input_node_name = mapper.outputs_info[input_unique_id]
        mapper._check_input(graph, input_node, input_node_name, node_outputs)
        graph.add_layer(
            "prim.assert",
            inputs={},
            outputs=[output_name + '_assert'],
            type='eq',
            key=mapper.attrs[input_node_name],
            value=1 if i == 1 else -1)
        flatten_inputs.append(input_node_name)
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.outputs_info[input_unique_id]
    mapper._check_input(graph, input_node, input_node_name, node_outputs)
    graph.add_layer(
        "fluid.layers.flatten",
        inputs={'x': input_node_name},
        outputs=[output_name],
        axis=1)
    flatten_inputs.append(input_node_name)
    return flatten_inputs, node_outputs


def aten___getitem__(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    inputs = {}
    for i, input_ivalue in enumerate(node.inputs()):
        input_node = input_ivalue.node()
        input_unique_id = input_ivalue.unique()
        input_node_name = mapper.outputs_info[input_unique_id]
        mapper._check_input(graph, input_node, input_node_name, node_outputs)
        if i == 0:
            inputs['list'] = input_node_name
        else:
            inputs['index'] = input_node_name
    graph.add_layer("prim.getitem", inputs=inputs, outputs=[output_name])
    return list(inputs.values()), node_outputs


def aten_le(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    inputs = {}
    for i, input_ivalue in enumerate(node.inputs()):
        input_node = input_ivalue.node()
        input_unique_id = input_ivalue.unique()
        input_node_name = mapper.outputs_info[input_unique_id]
        mapper._check_input(graph, input_node, input_node_name, node_outputs)
        inputs['input{}'.format(i)] = input_node_name
    graph.add_layer("prim.le", inputs=inputs, outputs=[output_name])
    return list(inputs.values()), node_outputs


def aten_len(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.outputs_info[input_unique_id]
    mapper._check_input(graph, input_node, input_node_name, node_outputs)
    graph.add_layer(
        "prim.len", inputs={'input': input_node_name}, outputs=[output_name])
    return [input_node_name], node_outputs


def aten_max_pool2d(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    inputs = {}
    attrs = {}
    pool_inputs = []
    if "pool" in mapper.dygraph_name_id:
        mapper.dygraph_name_id["pool"] += 1
    else:
        mapper.dygraph_name_id["pool"] = 0
    pool_name = "pool" + str(mapper.dygraph_name_id["pool"])
    for i, input_ivalue in enumerate(node.inputs()):
        input_node = input_ivalue.node()
        input_unique_id = input_ivalue.unique()
        input_node_name = mapper.outputs_info[input_unique_id]
        if i == 0:
            mapper._check_input(graph, input_node, input_node_name,
                                node_outputs)
            inputs['input'] = input_node_name
            pool_inputs.append(input_node_name)
        elif i == 1:
            attrs['pool_size'] = mapper.attrs[input_node_name]
        elif i == 2:
            attrs['pool_stride'] = mapper.attrs[input_node_name]
        elif i == 3:
            attrs['pool_padding'] = mapper.attrs[input_node_name]
        elif i == 4:
            graph.add_layer(
                "prim.assert",
                inputs={},
                outputs=[output_name + '_assert'],
                type='eq',
                key=mapper.attrs[input_node_name],
                value=[1, [1, 1]])
            pool_inputs.append(input_node_name)
        elif i == 5:
            attrs['ceil_mode'] = mapper.attrs[
                input_node_name] if input_node_name in mapper.attrs else input_node_name
            if input_node_name not in mapper.attrs:
                pool_inputs.append(input_node_name)
    attrs['pool_type'] = string('max')
    graph.add_layer(
        "fluid.dygraph.Pool2D",
        inputs=inputs,
        outputs=[pool_name, output_name],
        **attrs)
    return pool_inputs, node_outputs


def aten_matmul(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    inputs = {}
    x_node = list(node.inputs())[0].node()
    x_unique_id = list(node.inputs())[0].unique()
    x_node_name = mapper.outputs_info[x_unique_id]
    mapper._check_input(graph, x_node, x_node_name, node_outputs)
    inputs['x'] = x_node_name
    y_node = list(node.inputs())[1].node()
    y_unique_id = list(node.inputs())[1].unique()
    y_node_name = mapper.outputs_info[y_unique_id]
    inputs['y'] = y_node_name
    mapper._check_input(graph, y_node, y_node_name, node_outputs)
    graph.add_layer("fluid.layers.matmul", inputs=inputs, outputs=[output_name])
    return list(inputs.values()), node_outputs


def aten_relu_(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.outputs_info[input_unique_id]
    mapper._check_input(graph, input_node, input_node_name, node_outputs)
    # inplace这个参数在paddle中未实现
    graph.add_layer(
        "fluid.layers.relu",
        inputs={"x": input_node_name},
        outputs=[output_name])
    return [input_node_name], node_outputs


def aten_relu6(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.outputs_info[input_unique_id]
    mapper._check_input(graph, input_node, input_node_name, node_outputs)
    # inplace这个参数在paddle中未实现
    graph.add_layer(
        "fluid.layers.relu6",
        inputs={"x": input_node_name},
        outputs=[output_name],
        threshold=6.0)
    return [input_node_name], node_outputs


def aten_size(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.outputs_info[input_unique_id]
    mapper._check_input(graph, input_node, input_node_name, node_outputs)
    graph.add_layer(
        "prim.shape", inputs={'input': input_node_name}, outputs=[output_name])
    return [input_node_name], node_outputs


def aten_slice(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    attrs = {}
    slice_inputs = []
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.outputs_info[input_unique_id]
    slice_inputs.append(input_node_name)
    strat_node = list(node.inputs())[1].node()
    start_unique_id = list(node.inputs())[1].unique()
    start_node_name = mapper.outputs_info[start_unique_id]
    slice_inputs.append(start_node_name)
    attrs['start'] = mapper.attrs[
        start_node_name] if start_node_name in mapper.attrs else start_node_name
    if start_node_name not in mapper.attrs:
        mapper._check_input(graph, strat_node, start_node_name, node_outputs)
        slice_inputs.append(input_node_name)
    end_node = list(node.inputs())[2].node()
    end_unique_id = list(node.inputs())[2].unique()
    end_node_name = mapper.outputs_info[end_unique_id]
    slice_inputs.append(end_node_name)
    attrs['end'] = mapper.attrs[
        end_node_name] if end_node_name in mapper.attrs else end_node_name
    if end_node_name not in mapper.attrs:
        mapper._check_input(graph, end_node, end_node_name, node_outputs)
        slice_inputs.append(end_node_name)
    step_node = list(node.inputs())[3].node()
    step_unique_id = list(node.inputs())[3].unique()
    step_node_name = mapper.outputs_info[step_unique_id]
    slice_inputs.append(step_node_name)
    attrs['step'] = mapper.attrs[
        step_node_name] if step_node_name in mapper.attrs else step_node_name
    if step_node_name not in mapper.attrs:
        mapper._check_input(graph, step_node, step_node_name, node_outputs)
        slice_inputs.append(step_node_name)
    graph.add_layer(
        "prim.slice",
        inputs={'input': input_node_name},
        outputs=[output_name],
        **attrs)
    return [input_node_name], node_outputs


def aten_t(mapper, graph, node):
    output_name = mapper._get_outputs_name(node)[0]
    node_outputs = [output_name]
    input_node = list(node.inputs())[0].node()
    input_unique_id = list(node.inputs())[0].unique()
    input_node_name = mapper.outputs_info[input_unique_id]
    mapper._check_input(graph, input_node, input_node_name, node_outputs)
    graph.add_layer(
        "fluid.layers.transpose",
        inputs={"x": input_node_name},
        outputs=[output_name],
        perm=[1, 0])
    return [input_node_name], node_outputs
