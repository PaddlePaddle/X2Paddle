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
from x2paddle.optimizer.code_optimizer.parameter_tree import PamareterNode

NN_KERNEL_NAME = {"paddle.nn.BatchNorm": "bn",
                  "paddle.nn.LayerNorm": "layernorm",
                  "paddle.nn.Conv2d": "conv",
                  "paddle.fluid.dygraph.Embedding": "embedding",
                  "paddle.nn.Linear": "linear",
                  "paddle.nn.ReLU": "relu",
                  "paddle.nn.ReLU6": "relu",
                  "paddle.nn.Softmax": "softmax",
                  "paddle.nn.Softplus": "softplus",
                  "paddle.nn.Tanh": "tanh",
                  "paddle.nn.Pool2D": "pool",
                  "paddle.nn.ConstantPad1d": "constant_pad",
                  "paddle.nn.ConstantPad2d": "constant_pad",
                  "paddle.nn.ConstantPad3d": "constant_pad",
                  "paddle.nn.Dropout": "dropout",
                  "paddle.nn.GELU": "gelu",
                  "paddle.nn.Hardtanh": "tanh",
                  "paddle.nn.LeakyReLU": "leakly_relu"}
NN_KERNEL_WITH_PARAMS = list(NN_KERNEL_NAME.keys())[:5]

def rename_layers(layers, param_tree=None):
    """ 对子模块的输入输出等进行重命名。
    """
    layers_cp = copy.deepcopy(layers)
    name_dict = dict()
    nn_param_nodes = list()
    count = 0
    nn_count_dict = dict()
    module_count_dict = dict()
    new_names = list()
    for kernel in NN_KERNEL_NAME.keys():
        nn_count_dict[kernel] = 0
    def rename_sub_layers(sub_layers, count, is_block=False):
        for layer_id, layer in sub_layers.items():
            # 对输入重命名
            for input_k, input_v in layer.inputs.items():
                if input_v in name_dict:
                    layer.inputs[input_k] = name_dict[input_v]
                else:
                    new_name = "x{}".format(count)
                    count += 1
                    layer.inputs[input_k] = new_name
                    name_dict[input_v] = new_name
            # 对block重命名        
            for block in layer.blocks:
                count =  rename_sub_layers(block.layers, 
                                           count, is_block=True)
            # 对输出重命名
            if len(layer.outputs) == 0 and not is_block:
                new_names.append("layer_id/{}".format(layer_id))
            for i, output_v in enumerate(layer.outputs):
                if output_v in name_dict:
                    layer.outputs[i] = name_dict[output_v]
                    if i == 0 and not is_block:
                        new_names.append(name_dict[output_v])
                else:
                    if i == 0 and layer.kernel in NN_KERNEL_NAME.keys():
                        new_name = NN_KERNEL_NAME[layer.kernel] + str(nn_count_dict[layer.kernel])
                        param_node = PamareterNode(old_name=layer.outputs[0],
                                                   new_name=new_name)
                        nn_param_nodes.append(param_node)
                        if param_tree is not None:
                            param_tree.add_node(param_node)
                        layer.outputs[0] = new_name
                        nn_count_dict[layer.kernel] += 1
                    elif i == 0 and layer.kernel == "module":
                        old_name = layer.outputs[0].split("/")[0]
                        if old_name not in nn_count_dict:
                            nn_count_dict[old_name] = 0
                        else:
                            nn_count_dict[old_name] += 1
                        new_name = old_name + str(nn_count_dict[old_name])
                        if param_tree is not None:
                            param_node = param_tree.get_node(layer.outputs[0])
                            nn_param_nodes.append(param_node)
                            param_node.new_name = new_name
                        layer.outputs[0] = new_name
                    else:
                        new_name = "x{}".format(count)
                        count += 1
                        layer.outputs[i] = new_name
                        name_dict[output_v] = new_name
                    if i == 0 and not is_block:
                        new_names.append(new_name)
            # 对layer的attr进行重命名
            for attr_k, attr_v in layer.attrs.items():
                if isinstance(attr_v, str) and "'" not in attr_v \
                        and attr_v in name_dict:
                    layer.attrs[attr_k] = name_dict[attr_v]
        return count
    rename_sub_layers(layers_cp, count)
    return layers_cp, nn_param_nodes, new_names


def gen_layer_code(graph, sub_layers, sub_layers_name, different_attrs=list()):
    """ 根据sub_layers生成对应的Module代码。
    
    Args:
        graph (x2paddle.core.program.PaddleGraph): 整个Paddle图。
        sub_layers (dict): 子图的id和其对应layer组成的字典。
        sub_layers_name (str): 子图的名字。
        different_attrs (list): 属性列表，这些属性表明在被调用时赋予不同值。
    """
    def gen_codes(code_list, indent=0):
        """ 根据code_list生成代码段。
        
        Args:
            code_list (list): 代码行组成的list。
            indent (int): 每行空格的数量。
            
        Returns:
            str: 代码段。
        """
        indent_blank = "    " * indent
        codes = []
        for code_line in code_list:
            if code_line.strip() == "":
                codes.append('\n')
            else:
                codes.append(indent_blank + code_line + '\n')
        return codes
    
    def gen_head(inputs, different_attrs):
        # 生成Layer的头部代码
        head = gen_codes(["class {}(fluid.dygraph.Layer):".format(sub_layers_name)], indent=0)
        # 生成init函数的头部代码
        attrs_str = ", ".join(different_attrs)
        init_func_head = \
            gen_codes(["def __init__(self, params, {}):".format(attrs_str)], indent=1) + \
            gen_codes(["super({}, self).__init__()".format(sub_layers_name)], indent=2)
        # 生成forward函数的头部代码
        input_data_name = ", ".join(inputs)
        forward_func_head = \
            gen_codes(["def forward(self, {}):".format(input_data_name)], indent=1)
        return head, init_func_head, forward_func_head
        
    
    init_func = []
    forward_func = []
    cur_outputs = list()
    inputs = list()
    outputs = list()
    param_prefix_list = list()
    input_id = 0
    for layer_id, layer in sub_layers.items():
        if layer_id not in graph.edges_out:
            for output_name in layer.outputs:
                if not output_name.startswith("x") or output_name in outputs \
                        or layer.kernel == "prim.assert" or \
                        layer.kernel == "prim.if" or layer.kernel == "prim.loop":
                    continue
                elif output_name not in outputs:
                    outputs.append(output_name)
            continue
        for out_layer_id in graph.edges_out[layer_id]:
            if out_layer_id not in sub_layers:
                for output_name in layer.outputs:
                    if not output_name.startswith("x") or output_name in outputs \
                            or layer.kernel == "prim.assert" or \
                            layer.kernel == "prim.if" or layer.kernel == "prim.loop":
                        continue
                    else:
                        outputs.append(output_name)
    no_output_count = 0
    for i, (layer_id, layer) in enumerate(sub_layers.items()):
        if ("paddle.nn" in layer.kernel and "functional" not in layer.kernel
            ) or "paddle.fluid.dygraph" in layer.kernel or layer.kernel == "fluid.dygraph.base.to_variable":
            line = "self.{} = {}(".format(layer.outputs[0], layer.kernel)
            for k, v in layer.attrs.items():
                key_name = "{}_{}".format(layer.outputs[0], k)
                if key_name in different_attrs:
                    line += "{}={}, ".format(k, key_name)
                else:
                    line += "{}={}, ".format(k, v)
            line = line.strip(", ")
            line += ")"
            
            if layer.kernel == "fluid.dygraph.base.to_variable" and not layer.attrs[
                    "value"].startswith("params["):
                forward_func.extend(gen_codes([line.replace("self.", "")], indent=2))
                for k, v in layer.attrs.items():
                    if v not in cur_outputs:
                        inputs.append(v)
                    cur_outputs.append(layer.outputs[0])
                continue
            else:
                init_func.extend(gen_codes([line], indent=2))
            
            if len(layer.outputs) == 1:
                line = layer.outputs[0]
            elif len(layer.outputs) == 2:
                line = layer.outputs[1]
            else:
                line = ','.join(layer.outputs[1:])
            if layer.kernel == "fluid.dygraph.base.to_variable" and layer.attrs[
                        "value"].startswith("params["):
                line += " = self.{}".format(layer.outputs[0])
            else:
                line += " = self.{}(".format(layer.outputs[0])
                for k, v in layer.inputs.items():
                    if v not in cur_outputs and v not in inputs:
                        inputs.append(v)
                    line += "{}, ".format(v)
                line = line.strip(", ")
                line += ")"
            forward_func.extend(gen_codes([line], indent=2))
            if len(layer.outputs) == 1:
                cur_outputs.append(layer.outputs[0])
            else:
                cur_outputs.extend(layer.outputs[1:])
        elif "prim" in layer.kernel:
            func_name = layer.kernel.replace(".", "_")
            from x2paddle.op_mapper.pytorch2paddle import prim2code
            if hasattr(prim2code, func_name):
                for k, v in layer.inputs.items():
                    if v not in cur_outputs and v not in inputs:
                        inputs.append(v)
                func = getattr(prim2code, func_name)
                func(
                    layer,
                    indent=2,
                    init_func=init_func,
                    forward_func=forward_func,
                    layer_id=layer_id, 
                    different_attrs=different_attrs)
                cur_outputs.extend(layer.outputs)
            else:
                raise Exception(
                    "The kind {} in paddle model is not supported yet.".
                    format(layer.kernel))
        elif layer.kernel == "module":
            line = "self.{} = {}(".format(layer.outputs[0], layer.attrs["module"])
            layer.attrs.pop("module")
            for k, v in layer.attrs.items():
                key_name = "{}_{}".format(layer.outputs[0], k)
                if key_name in different_attrs:
                    line += "{}={}, ".format(k, key_name)
                else:
                    line += "{}={}, ".format(k, v)
            line = line.strip(", ")
            line += ")"
            init_func.extend(gen_codes([line], indent=2))
            if len(layer.outputs) == 2:
                line = layer.outputs[1]
            else:
                line = ','.join(layer.outputs[1:])
            line += " = self.{}(".format(layer.outputs[0])
            for k, v in layer.inputs.items():
                if v not in cur_outputs and v not in inputs:
                    inputs.append(v)
                line += "{}, ".format(v)
            line = line.strip(", ")
            line += ")"
            forward_func.extend(gen_codes([line], indent=2))
            cur_outputs.extend(layer.outputs[1:])
        else:
            if len(layer.outputs) == 1:
                line = layer.outputs[0]
            else:
                line = ','.join(layer.outputs)
            line += " = {}(".format(layer.kernel)
            for k, v in layer.inputs.items():
                if v not in cur_outputs and v not in inputs:
                    inputs.append(v)
                line += "{}={}, ".format(k, v)
            for k, v in layer.attrs.items():
                key_name = "{}_{}".format(layer.outputs[0], k)
                if key_name in different_attrs:
                    line += "{}=self.{}, ".format(k, key_name)
                    init_func.extend(gen_codes(["self.{} = {}".format(key_name, key_name)], indent=2))
                else:
                    line += "{}={}, ".format(k, v)
            line = line.strip(", ")
            line += ")"
            forward_func.extend(gen_codes([line], indent=2))
            cur_outputs.extend(layer.outputs)

    head, init_func_head, forward_func_head = gen_head(inputs, different_attrs)
    output_data_name  = ", ".join(outputs)
    code_list = head + init_func_head + init_func + \
                forward_func_head + forward_func + \
                gen_codes(["return {}".format(output_data_name)], indent=2)
    code_str = "".join(code_list)
    return code_str