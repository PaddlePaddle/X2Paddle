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

from x2paddle.core.graph import GraphNode
from x2paddle.core.util import *
import collections
import six


class Layer(object):
    def __init__(self):
        self.op = None
        self.param_attr = dict()
        self.inputs = dict()
        self.output = None
        self.is_custom_layer = False
        self.is_dygraph = False

    def get_code(self):
        layer_code = ""
        if self.output is not None:
            if isinstance(self.output, six.string_types):
                layer_code = self.output + " = "
            else:
                layer_code = self.output.layer_name + " = "

        if self.is_custom_layer:
            layer_code = layer_code + self.op + "("
        elif self.op == "=":
            layer_code = layer_code
        else:
            layer_code = layer_code + "fluid.layers." + self.op + "("

        if isinstance(self.inputs, list):
            in_list = "["
            for input in self.inputs:
                if isinstance(input, GraphNode):
                    if hasattr(input, "index"):
                        in_list += (
                            input.layer_name + "[{}]".format(input.index) + ", "
                        )
                    else:
                        in_list += (input.layer_name + ", ")
                elif isinstance(input, six.string_types):
                    in_list += (input + ", ")
                else:
                    raise Exception(
                        "Element of inputs should GraphNode or String")
            in_list = in_list.strip(", ") + "], "
            layer_code += in_list
        elif isinstance(self.inputs, dict):
            inputs = collections.OrderedDict(self.inputs)
            for key, input in inputs.items():
                if isinstance(input, GraphNode):
                    if hasattr(input, "index"):
                        layer_code = layer_code + key + "={}, ".format(
                            input.layer_name + "[{}]".format(input.index))
                    else:
                        layer_code = layer_code + key + "={}, ".format(
                            input.layer_name)
                else:
                    layer_code = layer_code + key + "={}, ".format(input)
        elif isinstance(self.inputs, GraphNode):
            if hasattr(self.inputs, "index"):
                layer_code += (
                    self.inputs.layer_name + "[{}]".format(self.inputs.index))
            else:
                layer_code += (self.inputs.layer_name)
            if self.op != "=":
                layer_code += ", "
        elif isinstance(self.inputs, six.string_types):
            layer_code += (self.inputs)
            if self.op != "=":
                layer_code += ", "
        else:
            raise Exception("Unknown type of inputs.")

        param_attr = collections.OrderedDict(self.param_attr)
        for key, value in param_attr.items():
            if '\n' in str(value):
                value = string(str(value).replace('\n', ','))
            if str(key) == 'attr':
                value = 'ParamAttr(' + str(value) + ')'
            layer_code = layer_code + key + "={}, ".format(value)
        layer_code = layer_code.strip(", ")

        if self.op != "=":
            layer_code += ")"
        return layer_code

    def get_code_dygraph(self):
        layer_code = ""
        if self.output is not None:
            assert isinstance(
                self.output,
                six.string_types), 'The output of dygraph node must be string.'
            layer_code = "self." + self.output + " = "
        layer_code = layer_code + "fluid.dygraph." + self.op + "("
        param_attr = collections.OrderedDict(self.param_attr)
        for key, value in param_attr.items():
            if '\n' in str(value):
                value = string(str(value).replace('\n', ','))
            layer_code = layer_code + key + "={}, ".format(value)
        layer_code = layer_code.strip(", ")
        layer_code += ")"
        return layer_code


class FluidCode(object):
    def __init__(self):
        self.layers = list()

    def add_layer(self,
                  op,
                  inputs,
                  output,
                  param_attr=None,
                  is_custom_layer=False):
        layer = Layer()
        layer.op = op
        layer.is_custom_layer = is_custom_layer
        if inputs is not None:
            layer.inputs = inputs
        layer.output = output
        if param_attr is not None:
            layer.param_attr = param_attr
        self.layers.append(layer)

    def add_note(self, note):
        # note should be string
        self.layers.append(note)

    def add_dygraph(self, op, name, inputs, output, param_attr=None):
        layer = Layer()
        layer.op = op
        layer.is_dygraph = True
        layer.output = name
        if inputs is not None:
            layer.inputs = inputs
        if param_attr is not None:
            layer.param_attr = param_attr
        self.layers.append(layer)
        input_node_names = []
        for input_node in inputs:
            if hasattr(input_node, 'index'):
                input_node_name = input_node.layer_name + "[{}]".format(
                    input_node.index)
            else:
                input_node_name = input_node.layer_name
            input_node_names.append(input_node_name)
        input_info = ','.join(input_node_names)
        if op == 'Dropout':
            self.add_note("self.{}.eval()".format(name))
        self.add_note("{} = {}({})".format(output.layer_name, "self." + name,
                                           input_info))

    def clear(self):
        self.layers = list()

    def gen_codes(self):
        codes = list()
        for layer in self.layers:
            if isinstance(layer, Layer):
                if layer.is_dygraph:
                    codes.append(layer.get_code_dygraph())
                else:
                    codes.append(layer.get_code())
            elif isinstance(layer, six.string_types):
                codes.append(layer)
        return codes