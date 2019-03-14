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

NHWC = 'NHWC'.encode()
NCHW = 'NCHW'.encode()
OTHER = 'OTHER'.encode()
SAME = 'SAME'.encode()
VALID = 'VALID'.encode()


class NameGenerator(object):
    def __init__(self):
        self.param_index = 0
        self.net_index = 0
        self.const_index = 0
        self.names = dict()

    def get_name(self, node):
        ref_name = None
        op_name = node.layer_type

        if node.layer.name in self.names:
            return self.names[node.layer.name]

        if op_name == "variablev2":
            ref_name = "param_" + str(self.param_index)
            self.param_index += 1
        elif op_name == "placeholder":
            ref_name = node.layer.name
        elif op_name == "const":
            ref_name = "const_" + str(self.const_index)
            self.const_index += 1
        elif op_name.lower() == "identity":
            ref_name = self.names[node.layer.input[0]]
        else:
            ref_name = "net_" + str(self.net_index)
            self.net_index += 1
        self.names[node.layer.name] = ref_name
        return ref_name


class LayerCode(object):
    def __init__(self):
        self.op = None
        self.param_attr = dict()
        self.input = None
        self.output = None
        self.str_code = None

    def get_str_code(self):
        if self.str_code is not None:
            return self.str_code

        layer_code0 = ""
        if self.output is not None:
            layer_code0 = layer_code0 + self.output + " = "
        layer_code0 += "layers."

        layer_code1 = self.op + "("
        if self.input is not None:
            layer_code1 = layer_code1 + self.input + ", "

        layer_code2 = ""
        for k, v in self.param_attr.items():
            layer_code2 = layer_code2 + k + "=" + "{}".format(v) + ", "
        layer_code2 = layer_code2.strip(", ")

        layer_code = (
            layer_code0 + layer_code1 + layer_code2).strip(", ") + ")"
        return layer_code


class FluidCode(object):
    def __init__(self):
        self.codes = list()

    def add_layer(self, op, input, output, param_attr=None):
        if param_attr is None:
            param_attr = dict()
        layer_code = LayerCode()
        layer_code.op = op
        layer_code.input = input
        layer_code.output = output
        layer_code.param_attr = param_attr
        self.codes.append(layer_code)

    def add_str(self, str_code):
        layer_code = LayerCode()
        layer_code.str_code = str_code
        self.codes.append(layer_code)

    def clear(self):
        self.codes = list()

    def gen_codes(self):
        res = list()
        if len(self.codes) == 0:
            return []
        for code in self.codes:
            if isinstance(code, LayerCode):
                res.append(code.get_str_code())
            else:
                raise Exception("Unexcept situation!")
        return res
