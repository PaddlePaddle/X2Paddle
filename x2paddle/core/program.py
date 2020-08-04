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

from __future__ import print_function
from __future__ import division
import collections
import os
import numpy
import six
import pickle


class PaddleLayer(object):
    def __init__(self, kernel, inputs, outputs, **kwargs):
        assert isinstance(
            inputs,
            dict), "parameter 'inputs' for PaddleLayer should be type of dict"
        assert isinstance(
            outputs,
            list), "parameter 'outputs' for PaddleLayer should be type of list"
        for k, v in inputs.items():
            if isinstance(v, list):
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
        self.attrs = kwargs
        self.blocks = []

    def add_block(self, block):
        self.blocks.append(block)


class PaddleGraph(object):
    def __init__(self):
        self.layers = list()
        self.edges_out = dict()
        self.edges_in = dict()
        self.inputs = list()
        self.outputs = list()
        self.parameters = dict()
        self.name = "PaddleNet"

    def set_name(self, name):
        self.name = name

    def clear(self):
        self.layers = list()
        self.edges_out = dict()
        self.edges_in = dict()
        self.inputs = list()
        self.outputs = list()
        self.parameters = dict()

    def add_layer(self, kernel, inputs, outputs, **kwargs):
        layer = PaddleLayer(kernel, inputs, outputs, **kwargs)
        index = len(self.layers)
        self.layers.append(layer)
        return index

    def build(self, inputs=None, outputs=None):
        outputs_from_nodes = dict()
        for i, layer in enumerate(self.layers):
            for input_key, input_var in layer.inputs.items():
                vs = input_var
                if not isinstance(vs, list):
                    vs = [vs]
                for v in vs:
                    assert v in outputs_from_nodes or (
                        inputs is not None and v in list(inputs.values())
                    ) or (
                        outputs is not None and v in outputs
                    ), "Couldn't find {} in previous layers, the layers should be make by topological sort".format(
                        v)
                    if v in outputs_from_nodes:
                        in_layer_index = outputs_from_nodes[v]
                    else:
                        in_layer_index = -1
                    if in_layer_index not in self.edges_out:
                        self.edges_out[in_layer_index] = list()
                    self.edges_out[in_layer_index].append(i)

                    if i not in self.edges_in:
                        self.edges_in[i] = list()
                    self.edges_in[i].append(in_layer_index)

            for output in layer.outputs:
                outputs_from_nodes[output] = i

            if len(layer.blocks) > 0:
                for block in layer.blocks:
                    block.build(layer.inputs, layer.outputs)

    def get_layer_outputs(self, i):
        return self.edges_out[i]

    def get_layer_inputs(self, i):
        return self.edges_in[i]

    def convert_prim(self, layer, indent=1):
        def gen_lines(code_list, indent=0):
            indent_blank = "    " * indent
            lines = []
            for code_line in code_list:
                if code_line.strip() == "":
                    lines.append('\n')
                else:
                    lines.append(indent_blank + code_line + '\n')
            return lines

        if layer.kernel == "prim.if":
            line = "if {} :".format(list(layer.inputs.values())[0])
            self.forward_lines.extend(gen_lines([line], indent=indent))
            block = layer.blocks[0]
            b_init_lines, b_forward_lines = block.gen_dygraph_code(
                indent=indent + 1)
            self.init_lines.extend(b_init_lines)
            self.forward_lines.extend(b_forward_lines)
            block = layer.blocks[1]
            if len(block.layers) > 0:
                line = "else:"
                self.forward_lines.extend(gen_lines([line], indent=indent))
                b_init_lines, b_forward_lines = block.gen_dygraph_code(
                    indent=indent + 1)
                self.init_lines.extend(b_init_lines)
                self.forward_lines.extend(b_forward_lines)
            return
        elif layer.kernel == "prim.loop":
            line = "for {} in range({}):".format(layer.outputs[1],
                                                 list(layer.inputs.values())[0])
            self.forward_lines.extend(gen_lines([line], indent=indent))
            block = layer.blocks[0]
            b_init_lines, b_forward_lines = block.gen_dygraph_code(
                indent=indent + 1)
            self.init_lines.extend(b_init_lines)
            self.forward_lines.extend(b_forward_lines)
            return
        elif layer.kernel == "prim.equal":
            line = "{} = {}".format(layer.outputs[0],
                                    list(layer.inputs.values())[0])
        elif layer.kernel == "prim.constant":
            line = "{} = {}".format(layer.outputs[0], layer.attrs["value"])
        elif layer.kernel == "prim.list":
            inputs_list = list(layer.inputs.values())
            inputs_str = ', '.join(inputs_list)
            line = "{} = [{}]".format(layer.outputs[0], inputs_str)
        elif layer.kernel == "prim.exception":
            line = "raise RaiseException({})".format(
                list(layer.inputs.values())[0])
        elif layer.kernel == "prim.min":
            line = "{} = min({})".format(layer.outputs[0],
                                         list(layer.inputs.values())[0])
        elif layer.kernel == "prim.add":
            line = "{} = {} + {} * {}".format(layer.outputs[0],
                                              list(layer.inputs.values())[0],
                                              layer.attrs["alpha"],
                                              list(layer.inputs.values())[1])
        elif layer.kernel == "prim.append":
            line = "{} = {}.append({})".format(layer.outputs[0],
                                               list(layer.inputs.values())[0],
                                               list(layer.inputs.values())[1])
        elif layer.kernel == "prim.shape":
            line = "{} = {}.shape".format(layer.outputs[0],
                                          list(layer.inputs.values())[0])
        elif layer.kernel == "prim.len":
            line = "{} = len({})".format(layer.outputs[0],
                                         list(layer.inputs.values())[0])
        elif layer.kernel == "prim.eq":
            line = "{} = {} == {}".format(layer.outputs[0],
                                          list(layer.inputs.values())[0],
                                          list(layer.inputs.values())[1])
        elif layer.kernel == "prim.assert":
            if layer.attrs["type"] == "eq":
                if isinstance(layer.attrs["value"], list):
                    s = ""
                    for v in layer.attrs["value"]:
                        s += "{} == {} or ".format(
                            list(layer.inputs.values())[0], v)
                    if len(s) > 0:
                        s = s[:-4]
                    line = "assert {}, \'The {} must be {}!\'".format(
                        s, list(layer.inputs.values())[0], layer.attrs["value"])
                else:
                    line = "assert {} == {}, \'The {} must be {}!\'".format(
                        list(layer.inputs.values())[0], layer.attrs["value"],
                        list(layer.inputs.values())[0], layer.attrs["value"])
            else:
                raise Exception("Not implement yet!")
        elif layer.kernel == "prim.getitem":
            line = "{} = {}[{}]".format(layer.outputs[0],
                                        list(layer.inputs.values())[0],
                                        list(layer.inputs.values())[1])
        elif layer.kernel == "prim.le":
            line = "{} = {} < {}".format(layer.outputs[0],
                                         list(layer.inputs.values())[0],
                                         list(layer.inputs.values())[1])
        elif layer.kernel == "prim.slice":
            attrs_str = ""
            for k, v in layer.attrs.items():
                attrs_str += "{}:".format(v)
            attrs_str = attrs_str[:-1]
            line = "{} = {}[{}]".format(layer.outputs[0],
                                        list(layer.inputs.values())[0],
                                        attrs_str)
        self.forward_lines.extend(gen_lines([line], indent=indent))
        return

    def export_paddle_params(self, code_dir, params, middle_numpy=None):
        params_output = open(os.path.join(code_dir, 'model.pdparams'), 'wb')
        pickle.dump(params, params_output)
        params_output.close()
        if middle_numpy is not None:
            middle_numpy_output = open(
                os.path.join(code_dir, 'middle_numpy.pkl'), 'wb')
            pickle.dump(middle_numpy, middle_numpy_output)
            middle_numpy_output.close()

    def get_dygraph_inputs(self, layers):
        for i, layer in enumerate(layers):
            if self.edges_in.get(i, 0) == 0 and self.edges_out.get(i, 0) == 0:
                continue
            if layer.kernel == "fluid.dygraph.base.to_variable":
                value = layer.attrs["value"]
                if not value.startswith("middle_numpy["):
                    self.inputs.append(value)
            if len(layer.blocks) > 0:
                for block in layer.blocks:
                    block.get_dygraph_inputs(block.layers)
                    self.inputs.extend(block.inputs)

    def get_dygraph_outputs(self, layers):
        for i, layer in enumerate(layers):
            if self.edges_in.get(i, 0) == 0 and self.edges_out.get(i, 0) == 0:
                continue
            if self.edges_out.get(i, 0) == 0:
                for output_name in layer.outputs:
                    if output_name.endswith("_assert"):
                        continue
                    self.outputs.append(output_name)
            if len(layer.blocks) > 0:
                for block in layer.blocks:
                    block.get_dygraph_outputs(block.layers)
                    self.outputs.extend(block.inputs)

    def gen_dygraph_code(self, code_dir=None, indent=2):
        def gen_lines(code_list, indent=0):
            indent_blank = "    " * indent
            lines = []
            for code_line in code_list:
                if code_line.strip() == "":
                    lines.append('\n')
                else:
                    lines.append(indent_blank + code_line + '\n')
            return lines

        self.init_lines = []
        # forward_func
        self.forward_lines = []
        # def gen_head
        if indent == 2 and code_dir is not None:
            start_lines = gen_lines(
                [
                    "from paddle.fluid.initializer import Constant",
                    "from paddle.fluid.param_attr import ParamAttr",
                    "import paddle.fluid as fluid",
                    "",
                    "class {}(fluid.dygraph.Layer):".format(self.name),
                ],
                indent=0)
            self.get_dygraph_inputs(self.layers)
            input_data_name = ', '.join(self.inputs)
            self.init_lines.extend(
                gen_lines(
                    ["def __init__(self, middle_numpy):"], indent=1))
            self.init_lines.extend(
                gen_lines(
                    ["super({}, self).__init__()".format(self.name)], indent=2))
            self.forward_lines.extend(
                gen_lines(
                    ["def forward(self, {}):".format(input_data_name)],
                    indent=1))

        for i, layer in enumerate(self.layers):
            if self.edges_in.get(i, 0) == 0 and self.edges_out.get(i, 0) == 0:
                continue
            if "dygraph" in layer.kernel:
                line = "{}".format(
                    layer.outputs[0]
                ) if layer.kernel == "fluid.dygraph.base.to_variable" and not layer.attrs[
                    "value"].startswith("middle_numpy[") else "self.{}".format(
                        layer.outputs[0])
                line += " = {}(".format(layer.kernel)
                for k, v in layer.attrs.items():
                    line += "{}={}, ".format(k, v)
                line = line.strip(", ")
                line += ")"

                if layer.kernel == "fluid.dygraph.base.to_variable" and not layer.attrs[
                        "value"].startswith("middle_numpy["):
                    self.forward_lines.extend(gen_lines([line], indent=indent))
                    continue
                else:
                    self.init_lines.extend(gen_lines([line], indent=2))

                if len(layer.outputs) == 1:
                    line = layer.outputs[0]
                elif len(layer.outputs) == 2:
                    line = layer.outputs[1]
                else:
                    line = ','.join(layer.outputs[1:])
                if layer.kernel == "fluid.dygraph.base.to_variable" and layer.attrs[
                        "value"].startswith("middle_numpy["):
                    line += " = self.{}".format(layer.outputs[0])
                else:
                    line += " = self.{}(".format(layer.outputs[0])
                    for k, v in layer.inputs.items():
                        line += "{}, ".format(v)
                    line = line.strip(", ")
                    line += ")"
                self.forward_lines.extend(gen_lines([line], indent=indent))
            elif "prim" in layer.kernel:
                self.convert_prim(layer, indent=indent)
            else:
                if len(layer.outputs) == 1:
                    line = layer.outputs[0]
                else:
                    line = ','.join(layer.outputs)
                line += " = {}(".format(layer.kernel)
                for k, v in layer.inputs.items():
                    line += "{}={}, ".format(k, v)
                for k, v in layer.attrs.items():
                    line += "{}={}, ".format(k, v)
                line = line.strip(", ")
                line += ")"
                self.forward_lines.extend(gen_lines([line], indent=indent))
        if indent == 2:
            f = open(os.path.join(code_dir, 'code.py'), 'w')
            for line in start_lines:
                f.write(line)
            init_writen_line = []
            for line in self.init_lines:
                if line in init_writen_line:
                    continue
                f.write(line)
                init_writen_line.append(line)
            f.write("\n")
            self.get_dygraph_outputs(self.layers)
            return_line = "return {}".format(", ".join(self.outputs))
            self.forward_lines.extend(gen_lines([return_line], indent=2))
            for line in self.forward_lines:
                f.write(line)
            f.close()
        else:
            return self.init_lines, self.forward_lines

    def gen_code(self, code_dir):
        def write_code(f, code_list, indent=0):
            indent_blank = "    " * indent
            for code_line in code_list:
                if code_line.strip() == "":
                    f.write('\n')
                else:
                    f.write(indent_blank + code_line + '\n')

        if not os.path.exists(code_dir):
            os.makedirs(code_dir)
        f = open(os.path.join(code_dir, 'x2paddle_model.py'), 'w')

        write_code(
            f, [
                "from paddle.fluid.initializer import Constant",
                "from paddle.fluid.param_attr import ParamAttr",
                "import paddle.fluid as fluid"
                "", "def x2paddle_net():"
            ],
            indent=0)
        for i, layer in enumerate(self.layers):
            edges_in = self.edges_in.get(i, [])
            edges_out = self.edges_out.get(i, [])
            if len(edges_in) == 0 and len(edges_out) == 0:
                continue

            line = ""

            if len(layer.outputs) == 1:
                line = layer.outputs[0]
            else:
                for output in layer.outputs:
                    line += "{}, ".format(output)
                line = line.strip(", ")

            line += " = {}(".format(layer.kernel)
            for k, v in layer.inputs.items():
                if isinstance(v, list):
                    line += "{}=[{}], ".format(k, ", ".join(v))
                else:
                    line += "{}={}, ".format(k, v)
            for k, v in layer.attrs.items():
                line += "{}={}, ".format(k, v)
            line = line.strip(", ")
            line += ")"
            write_code(f, [line], indent=1)

        write_code(
            f, [
                "return [{}], [{}]".format(", ".join(self.inputs),
                                           ", ".join(self.outputs))
            ],
            indent=1)
        f.close()

    def gen_parameters(self, code_dir):
        pass

    def gen_inference_model(self, model_dir):
        pass
