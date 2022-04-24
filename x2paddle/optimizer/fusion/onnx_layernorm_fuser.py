#   Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict
from x2paddle.optimizer.pattern_matcher import FuseBase
from x2paddle.core.program import PaddleGraph, PaddleLayer
from x2paddle.core.util import *


class LayerNormFuser(FuseBase):
    def __init__(self):
        super(LayerNormFuser, self).__init__()

    def build_pattern(self):
        """
        code describe:
            x2paddle_ln_pre_weight = self.x2paddle_ln_pre_weight
            x2paddle_ln_pre_bias = self.x2paddle_ln_pre_bias
            x2paddle_166 = paddle.full(dtype='float32', shape=[1], fill_value=2.0)
            x2paddle_169 = paddle.full(dtype='float32', shape=[1], fill_value=9.999999747378752e-06)
            x2paddle_164 = paddle.mean(x=x2paddle_162, axis=[-1], keepdim=True)
            x2paddle_165 = paddle.subtract(x=x2paddle_162, y=x2paddle_164)
            x2paddle_167 = paddle.pow(x=x2paddle_165, y=x2paddle_166)
            x2paddle_168 = paddle.mean(x=x2paddle_167, axis=[-1], keepdim=True)
            x2paddle_170 = paddle.add(x=x2paddle_168, y=x2paddle_169)
            x2paddle_171 = paddle.sqrt(x=x2paddle_170)
            x2paddle_172 = paddle.divide(x=x2paddle_165, y=x2paddle_171)
            x2paddle_173 = paddle.multiply(x=x2paddle_172, y=x2paddle_ln_pre_weight)
            x2paddle_174 = paddle.add(x=x2paddle_173, y=x2paddle_ln_pre_bias)
        """

        def gen_name(id):
            return "x" + str(id)

        self.pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(0)])
        self.pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(1)])
        self.pattern.add_layer(
            "paddle.full",
            inputs={},
            outputs=[gen_name(2)],
            shape=[1],
            fill_value=0.5)
        self.pattern.add_layer(
            "paddle.full", inputs={}, outputs=[gen_name(3)], shape=[1])
        self.pattern.add_layer(
            "paddle.mean",
            inputs={"x": "layernorm-input-0"},
            outputs=[gen_name(4)],
            axis=[-1],
            keep_dim=True)
        self.pattern.add_layer(
            "paddle.subtract",
            inputs={"x": "layernorm-input-0",
                    "y": gen_name(4)},
            outputs=[gen_name(5)])
        self.pattern.add_layer(
            "paddle.pow",
            inputs={"x": gen_name(5),
                    "y": gen_name(2)},
            outputs=[gen_name(6)])
        self.pattern.add_layer(
            "paddle.mean",
            inputs={"x": gen_name(6)},
            outputs=[gen_name(7)],
            axis=[-1],
            keep_dim=True)
        self.pattern.add_layer(
            "paddle.add",
            inputs={"x": gen_name(7),
                    "y": gen_name(3)},
            outputs=[gen_name(8)])
        self.pattern.add_layer(
            "paddle.sqrt", inputs={"x": gen_name(8)}, outputs=[gen_name(9)])
        self.pattern.add_layer(
            "paddle.divide",
            inputs={"x": gen_name(5),
                    "y": gen_name(9)},
            outputs=[gen_name(10)])
        self.pattern.add_layer(
            "paddle.multiply",
            inputs={"x": gen_name(10),
                    "y": gen_name(0)},
            outputs=[gen_name(11)])
        self.pattern.add_layer(
            "paddle.add",
            inputs={"x": gen_name(11),
                    "y": gen_name(1)},
            outputs=[gen_name(12)])
        self.pattern.build(inputs={"input-0": "layernorm-input-0", })

    def insert_new_layer(self, graph, parameters, matches):
        new_layer, new_layer_id = self.gen_new_layer(parameters, matches)
        graph.layers[new_layer_id] = new_layer
        matches_copy = copy.deepcopy(matches)
        for layer_id, layer in matches_copy.items():
            if layer.kernel in ["self.create_parameter", "paddle.full"]:
                matches.pop(layer_id)
        matches.pop(new_layer_id)

    def gen_new_layer(self, parameters, matches):
        layer_id_list = list(matches.keys())
        layer_id_list.sort(key=int)
        layer_inputs = list()
        layer_inputs_ids = list()
        param_name = list()
        fill_value_list = list()
        for layer_id, layer in matches.items():
            if layer.kernel == "paddle.mean":
                layer_inputs.append(layer.inputs)
                layer_inputs_ids.append(layer_id)
            if layer.kernel == "self.create_parameter":
                param_name.append(layer.outputs[0])
            if layer.kernel == "paddle.add":
                output_name = layer.outputs[0]
            if layer.kernel == "paddle.full":
                fill_value_list.append(layer.attrs["fill_value"])
        param = parameters[param_name[0]]
        c = param.shape[0]
        weight_param = parameters.pop(param_name[0])
        parameters["{}.weight".format(output_name)] = weight_param
        bias_param = parameters.pop(param_name[1])
        parameters["{}.bias".format(output_name)] = bias_param
        new_layer = PaddleLayer(
            layer_id_list[0],
            "paddle.nn.LayerNorm",
            inputs=layer_inputs[0],
            outputs=[output_name],
            normalized_shape=[c],
            epsilon=fill_value_list[-1])
        return new_layer, layer_inputs_ids[0]
