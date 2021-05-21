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

from __future__ import print_function
from __future__ import division
import paddle
import collections
import sys
import os
import six
import pickle
from os import path as osp
from x2paddle.core.util import *


class PaddleLayer(object):
    def __init__(self, id, kernel, inputs, outputs, scope_name="", **kwargs):
        assert isinstance(
            inputs,
            dict), "parameter 'inputs' for PaddleLayer should be type of dict"
        assert isinstance(
            outputs,
            list), "parameter 'outputs' for PaddleLayer should be type of list"
        for k, v in inputs.items():
            if isinstance(v, (list, tuple)):
                for i in v:
                    assert isinstance(
                        i, six.string_types
                    ), "value in inputs should be type of string or list of string"
            else:
                assert isinstance(v, six.string_types) or isinstance(
                    v, list
                ), "value in inputs should be type of string or list of string"
        for v in outputs:
            assert isinstance(
                v, six.
                string_types), "elements in outputs should be type of string"
        self.kernel = kernel
        self.inputs = inputs
        self.outputs = outputs
        self.scope_name = scope_name
        self.attrs = kwargs
        self.id = id
        self.blocks = list()

    def add_block(self, block):
        self.blocks.append(block)


class PaddleGraph(object):
    def __init__(self, source_type=None, parent_layer=None):
        self.layers = collections.OrderedDict()
        self.edges_out = dict()
        self.edges_in = dict()
        self.inputs = list()
        self.outputs = list()
        self.parameters = dict()
        self.parent_layer = parent_layer
        self.source_type = source_type
        self.custom_code = None
        self.inputs_info = None
        self.has_unpack = False

    def set_name(self, name):
        self.name = name.replace("-", "_").replace("/", "_")

    def set_parameters(self, parameters):
        self.parameters = parameters

    def set_custom(self, custom_code):
        self.custom_code = custom_code

    def set_inputs_info(self, inputs_info):
        self.inputs_info = inputs_info

    def set_script(self, script):
        self.script = script

    def clear(self):
        self.layers = collections.OrderedDict()
        self.edges_out = dict()
        self.edges_in = dict()
        self.inputs = list()
        self.outputs = list()
        self.parameters = dict()

    def clear_edges(self):
        self.edges_out = dict()
        self.edges_in = dict()

    def add_layer(self, kernel, inputs, outputs, scope_name="", **kwargs):
        layer_id = str(len(self.layers))
        if self.parent_layer is not None:
            layer_id = "{}.{}.{}".format(self.parent_layer.id,
                                         len(self.parent_layer.blocks),
                                         layer_id)
        layer = PaddleLayer(
            layer_id, kernel, inputs, outputs, scope_name=scope_name, **kwargs)
        self.layers[layer_id] = layer
        if layer.kernel in ["prim.list_unpack" or "prim.tuple_unpack"]:
            self.has_unpack = True
        return layer_id

    def del_layer(self, layer_id):
        layer = self.layers[layer_id]
        outputs = self.edges_out.get(layer_id, [])
        inputs = self.edges_in.get(layer_id, [])

        assert len(
            inputs) <= 1, "There should be 0 or 1 input for deleted layer."

        if len(inputs) == 0:
            for out in outputs:
                while layer_id in self.edges_in[out]:
                    index = self.edges_in[out].index(layer_id)
                    del self.edges_in[out][index]

                input_keys = list(self.layers[out].inputs.keys())
                for k in input_keys:
                    if self.layers[out].inputs[k] == layer.outputs[0]:
                        del self.layers[out].inputs[k]

            del self.layers[layer_id]
            if layer_id in self.edges_in:
                del self.edges_in[layer_id]
            if layer_id in self.edges_out:
                del self.edges_out[layer_id]
            return

        # 将所有输出layer的输入layer进行替换
        for out in outputs:
            for i in range(len(self.edges_in[out])):
                if self.edges_in[out][i] == layer_id:
                    self.edges_in[out][i] = inputs[0]

        # 将输出layer赋给输入layer的输出
        replace_index = self.edges_out[inputs[0]].index(layer_id)
        del self.edges_out[inputs[0]][replace_index]
        for i, out in enumerate(outputs):
            self.edges_out[inputs[0]].insert(replace_index + i, out)
            for k, v in self.layers[out].inputs.items():
                if v == layer.outputs[0]:
                    self.layers[out].inputs[k] = list(layer.inputs.values())[0]

        del self.layers[layer_id]
        if layer_id in self.edges_out:
            del self.edges_out[layer_id]
        if layer_id in self.edges_in:
            del self.edges_in[layer_id]

    def build(self, inputs=None, outputs=None):
        self.clear_edges()
        outputs_from_nodes = dict()
        for layer_id, layer in self.layers.items():
            for input_key, input_var in layer.inputs.items():
                vs = input_var
                if not isinstance(vs, (list, tuple)):
                    vs = [vs]
                for v in vs:
                    assert v in outputs_from_nodes or (
                        inputs is not None and v in list(inputs.values())
                    ) or (
                        outputs is not None and v in outputs
                    ), "Couldn't find {} in previous layers, the layers should be make by topological sort".format(
                        v)
                    if v in outputs_from_nodes:
                        in_layer_id = outputs_from_nodes[v]
                    else:
                        in_layer_id = -1
                    if in_layer_id not in self.edges_out:
                        self.edges_out[in_layer_id] = list()
                    self.edges_out[in_layer_id].append(layer_id)

                    if layer_id not in self.edges_in:
                        self.edges_in[layer_id] = list()
                    self.edges_in[layer_id].append(in_layer_id)
            for output in layer.outputs:
                outputs_from_nodes[output] = layer_id

            # 将block的输出用于父图
            if inputs is not None and outputs is not None and set(
                    layer.outputs).issubset(outputs):
                if layer_id not in self.edges_out:
                    self.edges_out[layer_id] = list()
                self.edges_out[layer_id].append(-1)

            # 处理子图
            if len(layer.blocks) > 0:
                for block in layer.blocks:
                    block.build(layer.inputs, layer.outputs)

        # 删除不必要的节点
        invalid_list = list()
        for layer_id, layer in self.layers.items():
            if len(self.layers) > 1:
                if self.edges_in.get(layer_id, 0) == 0 and self.edges_out.get(
                        layer_id, 0) == 0 and layer.kernel != "prim.assert" \
                        and layer.kernel != "prim.exception" \
                        and layer.kernel != "prim.warnings" \
                        and layer.outputs[0] not in self.outputs:
                    if layer.kernel == "paddle.to_tensor" and layer.outputs[
                            0] in self.inputs_info:
                        self.inputs_info.pop(layer.outputs[0])
                    if layer.outputs[0] in self.inputs:
                        self.inputs.pop(self.inputs.index(layer.outputs[0]))
                    invalid_list.append(layer_id)
        for layer_id in invalid_list:
            self.layers.pop(layer_id)

        self.get_inputs()
        if len(self.outputs) == 0:
            self.get_outputs()

    def get_global_layers(self):
        # 该全局layers的信息是按照拓扑排序组成的
        def update(layers):
            global_layers = dict()
            for layer_id, layer in layers.items():
                global_layers[layer_id] = layer
                for block in layer.blocks:
                    block_global_layers = update(block.layers)
                    global_layers.update(block_global_layers)
            return global_layers

        return update(self.layers)

    def gen_model(self, save_dir, jit_type=None):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if jit_type == "trace":
            if not self.has_unpack:
                from x2paddle.optimizer.pytorch_code_optimizer import HierarchicalTree
                hierarchical_tree = HierarchicalTree(self)
                for layer_id, layer in self.layers.items():
                    hierarchical_tree.insert(layer)
                hierarchical_tree.save_source_files(save_dir)
                self.dump_parameter(save_dir)
            else:
                self.gen_code(save_dir)
                self.dump_parameter(save_dir)
        else:
            if self.source_type == "pytorch":
                from x2paddle.optimizer.pytorch_code_optimizer import ModuleGraph
                module_graph = ModuleGraph(self)
                module_graph.save_source_files(save_dir)
                self.dump_parameter(save_dir)
            else:
                self.gen_code(save_dir)
                self.dump_parameter(save_dir)
        # 动转静
        code_path = osp.join(osp.abspath(save_dir), "x2paddle_code.py")
        print("Exporting inference model from python code ('{}')... \n".format(
            code_path))
        if len(self.inputs_info) > 0:
            input_shapes = list()
            input_types = list()
            for input_name in self.inputs:
                input_shapes.append(self.inputs_info[input_name][0])
                input_types.append(self.inputs_info[input_name][1])
            try:
                self.dygraph2static(save_dir, input_shapes, input_types)
            except Exception as e:
                print(
                    "Fail to generate inference model! Problem happend while export inference model from python code '{}';\n".
                    format(code_path))
                print("===================Error Information===============")
                raise e

    def get_inputs(self):
        def update(layers):
            for layer_id, layer in layers.items():
                if self.edges_in.get(layer_id, 0) == 0 and self.edges_out.get(
                        layer_id, 0) == 0:
                    continue
                if layer.kernel == "paddle.to_tensor":
                    data = layer.attrs["data"]
                    self.inputs.append(data)
                if len(layer.blocks) > 0:
                    for block in layer.blocks:
                        block.get_inputs()
                        self.inputs.extend(block.inputs)

        update(self.layers)
        self.inputs = list(set(self.inputs))
        if self.inputs is not None:
            self.inputs.sort()

    def get_outputs(self):
        for layer_id, layer in self.layers.items():
            if self.edges_in.get(layer_id, 0) == 0 and self.edges_out.get(
                    layer_id, 0) == 0:
                continue
            if self.edges_out.get(layer_id, 0) == 0:

                for i, output_name in enumerate(layer.outputs):
                    if ("paddle.nn" in layer.kernel and
                            "functional" not in layer.kernel):
                        if i == 0:
                            continue
                    if output_name not in self.outputs:
                        self.outputs.append(output_name)

    def gen_code(self, code_dir=None, indent=2):
        def gen_codes(code_list, indent=0):
            indent_blank = "    " * indent
            codes = []
            for code_line in code_list:
                if code_line.strip() == "":
                    codes.append('\n')
                else:
                    codes.append(indent_blank + code_line + '\n')
            return codes

        def gen_head():
            if self.source_type == "caffe":
                custom_import = "from x2paddle.op_mapper.caffe2paddle " + \
                                 "import caffe_custom_layer as x2paddle_nn"
            elif self.source_type == "pytorch":
                custom_import = "from x2paddle.op_mapper.pytorch2paddle " + \
                                 "import pytorch_custom_layer as x2paddle_nn"
            elif self.source_type == "onnx":
                custom_import = "from x2paddle.op_mapper.onnx2paddle " + \
                                 "import onnx_custom_layer as x2paddle_nn"
            else:
                custom_import = ""
            self.head = gen_codes(
                [
                    "import paddle",
                    "import math",
                    custom_import,
                    "",
                    "class {}(paddle.nn.Layer):".format(self.name),
                ],
                indent=0)
            input_data_name = ', '.join(self.inputs)
            self.init_func.extend(gen_codes(["def __init__(self):"], indent=1))
            self.init_func.extend(
                gen_codes(
                    ["super({}, self).__init__()".format(self.name)], indent=2))
            self.forward_func.extend(
                gen_codes(
                    ["def forward(self, {}):".format(input_data_name)],
                    indent=1))

        def gen_main_code(code_dir):
            input_data_name = ', '.join(self.inputs)
            self.run_func = gen_codes(
                [
                    "",
                    "def main({}):".format(input_data_name),
                ], indent=0)
            comment_list = list()
            comment_list.append("# There are {} inputs.".format(
                len(self.inputs_info)))
            for k, v in self.inputs_info.items():
                comment_list.append("# {}: shape-{}, type-{}.".format(k, v[0],
                                                                      v[1]))
            self.run_func.extend(gen_codes(comment_list, indent=1))
            use_structured_name = False if self.source_type in ["tf"] else True
            self.run_func.extend(
                gen_codes(
                    [
                        "paddle.disable_static()",
                        "params = paddle.load('{}')".format(
                            osp.join(osp.abspath(code_dir), "model.pdparams")),
                        "model = {}()".format(self.name),
                        "model.set_dict(params, use_structured_name={})".format(
                            use_structured_name), "model.eval()",
                        "out = model({})".format(input_data_name), "return out"
                    ],
                    indent=1))

        def write_code(code_dir):
            f = open(osp.join(code_dir, 'x2paddle_code.py'), 'w')
            for code_line in self.head:
                f.write(code_line)
            init_writen_codes = []
            for code_line in self.init_func:
                if code_line in init_writen_codes:
                    continue
                f.write(code_line)
                init_writen_codes.append(code_line)
            f.write("\n")
            return_code = "return {}".format(", ".join(self.outputs))
            self.forward_func.extend(gen_codes([return_code], indent=2))
            for code_line in self.forward_func:
                if "assert [1, 1] == 1 or [1, 1] == [1, 1], 'The [1, 1] must be [1, [1, 1]]!'" in code_line:
                    continue
                f.write(code_line)
            for code_line in self.run_func:
                f.write(code_line)
            f.close()

        self.init_func = []
        self.forward_func = []
        if indent == 2 and code_dir is not None:
            gen_head()

        for layer_id, layer in self.layers.items():
            if layer.kernel.startswith("paddle"):
                remove_default_attrs(layer.kernel, layer.attrs)
            if ("paddle.nn" in layer.kernel and "functional" not in layer.kernel
                ) or layer.kernel == "paddle.to_tensor" or \
                layer.kernel.startswith("custom_layer"):
                line = "{}".format(
                    layer.outputs[0]
                ) if layer.kernel == "paddle.to_tensor" and not layer.attrs[
                    "data"].startswith("params[") else "self.{}".format(
                        layer.outputs[0])
                if layer.kernel.startswith("custom_layer"):
                    line += "= x2paddle_nn.{}(".format(
                        layer.kernel.split(":")[-1])
                else:
                    line += " = {}(".format(layer.kernel)
                for k, v in layer.attrs.items():
                    line += "{}={}, ".format(k, v)
                line = line.strip(", ")
                line += ")"

                if layer.kernel == "paddle.to_tensor" and not layer.attrs[
                        "data"].startswith("params["):
                    self.forward_func.extend(gen_codes([line], indent=indent))
                    continue
                else:
                    self.init_func.extend(gen_codes([line], indent=2))

                if len(layer.outputs) == 1:
                    line = layer.outputs[0]
                elif len(layer.outputs) == 2:
                    line = layer.outputs[1]
                else:
                    if layer.kernel in ["paddle.nn.LSTM"]:
                        line = "{}, ({})".format(layer.outputs[1],
                                                 ', '.join(layer.outputs[-2:]))
                    else:
                        line = ','.join(layer.outputs[1:])
                if layer.kernel == "paddle.to_tensor" and layer.attrs[
                        "data"].startswith("params["):
                    line += " = self.{}".format(layer.outputs[0])
                else:
                    line += " = self.{}(".format(layer.outputs[0])
                    for v in layer.inputs.values():
                        if isinstance(v, list):
                            line += "[{}], ".format(", ".join(v))
                        elif isinstance(v, tuple):
                            line += "({}), ".format(", ".join(v))
                        else:
                            line += "{}, ".format(v)
                    line = line.strip(", ")
                    line += ")"
                self.forward_func.extend(gen_codes([line], indent=indent))
            elif "prim" in layer.kernel:
                func_name = layer.kernel.replace(".", "_")
                from x2paddle.op_mapper.pytorch2paddle import prim2code
                if hasattr(prim2code, func_name):
                    func = getattr(prim2code, func_name)
                    func(
                        layer,
                        indent=indent,
                        init_func=self.init_func,
                        forward_func=self.forward_func)
                else:
                    raise Exception(
                        "The kind {} in paddle model is not supported yet.".
                        format(layer.kernel))
            else:
                if len(layer.outputs) == 1:
                    line = layer.outputs[0]
                else:
                    line = ','.join(layer.outputs)
                line += " = {}(".format(layer.kernel)
                for k, v in layer.inputs.items():
                    if isinstance(v, list):
                        line += "{}=[{}], ".format(k, ", ".join(v))
                    elif isinstance(v, tuple):
                        line += "{}=({}), ".format(k, ", ".join(v))
                    else:
                        if k == "args":
                            line += v
                        else:
                            line += "{}={}, ".format(k, v)
                for k, v in layer.attrs.items():
                    line += "{}={}, ".format(k, v)
                line = line.strip(", ")
                line += ")"
                if layer.kernel == "self.create_parameter":
                    self.init_func.extend(gen_codes(["self." + line], indent=2))
                    self.forward_func.extend(
                        gen_codes(
                            [
                                "{} = self.{}".format(layer.outputs[0],
                                                      layer.outputs[0])
                            ],
                            indent=indent))
                else:
                    self.forward_func.extend(gen_codes([line], indent=indent))
        if indent == 2 and code_dir is not None:
            gen_main_code(code_dir)
            write_code(code_dir)
        else:
            return self.init_func, self.forward_func

    def dump_parameter(self, code_dir):
        save_path = osp.join(code_dir, 'model.pdparams')
        paddle.save(self.parameters, save_path)

    def dygraph2static(self, save_dir, input_shapes=[], input_types=[]):
        sepc_list = list()
        for i, name in enumerate(self.inputs):
            sepc_list.append(
                paddle.static.InputSpec(
                    shape=input_shapes[i], name=name, dtype=input_types[i]))
        path = osp.abspath(save_dir)
        sys.path.insert(0, save_dir)
        import x2paddle_code
        paddle.disable_static()
        restore = paddle.load(osp.join(save_dir, "model.pdparams"))
        model = getattr(x2paddle_code, self.name)()
        if self.source_type in ["tf"]:
            model.set_dict(restore, use_structured_name=False)
        else:
            model.set_dict(restore)
        model.eval()
        static_model = paddle.jit.to_static(model, input_spec=sepc_list)
        try:
            paddle.jit.save(static_model,
                            osp.join(save_dir, "inference_model/model"))
        except ValueError as e:
            if str(e) == "'target_vars' should be a list of Variable.":
                print(
                    "[DyGraph2StaticGraph Error] Can not convert the dygraph to static! The output of PyTorch mustbe Variable or a list of Variable."
                )
            else:
                print(e)
                exit(0)
