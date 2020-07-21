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


class PaddleLayer(object):
    def __init__(self, kernel, inputs, outputs, **kwargs):
        assert isinstance(
            inputs,
            dict), "parameter 'inputs' for PaddleLayer should be type of dict"
        assert isinstance(
            outputs,
            list), "parameter, 'outputs' for PaddleLayer should be type of list"
        self.kernel = kernel
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = kwargs


class PaddleProgram(object):
    def __init__(self):
        self.layers = list()
        self.edges_out = dict()
        self.edges_in = dict()
        self.inputs = list()
        self.outputs = list()
        self.parameters = dict()

    def add_layer(self, kernel, inputs, outputs, **kwargs):
        layer = PaddleLayer(kernel, inputs, outputs, **kwargs)
        self.layers.append(layer)

    def build(self):
        outputs = dict()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            for output in layer.outputs:
                outputs[output] = i

            for k, v in layer.inputs.items():
                assert v in outputs, "Couldn't find {} in previous layers, the layers should be make by topological sort".format(
                    v)
                in_layer_index = outputs[v]

                if in_layer_index not in self.edges_out:
                    self.edges_out[in_layer_index] = list()
                self.edges_out[in_layer_index].append(i)

                if i not in self.edges_in:
                    self.edges_in[i] = list()
                self.edges_in[i].append(in_layer_index)

    def get_layer_outputs(self, i):
        return self.edges_out[i]

    def get_layer_inputs(self, i):
        return self.edges_in[i]

    def gen_code(self, code_dir):
        def write_code(f, code_list, indent=0):
            indent_blank = "    " * indent
            for code_line in code_list:
                if code_line.strip() == "":
                    f.write('\n')
                else:
                    f.write(indent_blank + code_line + '\n')

        f = open(os.path.join(code_dir, 'model.py'), 'w')

        write_code(
            f, [
                "from paddle.fluid.initializer import Constant",
                "from paddle.fluid.param_attr import ParamAttr",
                "import paddle.fluid as fluid"
                "", "def x2paddle_net():"
            ],
            indent=0)

        for i, layer in enumerate(self.layers):
            if self.edges_in.get(i, 0) == 0 and self.edges_out.get(i, 0) == 0:
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
                line += "{}={}, ".format(k, v)
            for k, v in layer.attrs.items():
                line += "{}={}, ".format(k, v)
            line = line.strip(", ")
            line += ")"
            write_code(f, [line], indent=1)
        f.close()

    def gen_parameters(self, code_dir):
        pass

    def gen_inference_model(self, model_dir):
        pass
