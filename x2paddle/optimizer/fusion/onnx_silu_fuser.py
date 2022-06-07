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


class SiluFuser(FuseBase):
    def __init__(self):
        super(SiluFuser, self).__init__()
        self.silu_index = 0

    def build_pattern(self):
        """
        code describe:
            x2paddle_123 = self.sigmoid0(x2paddle_122)
            x2paddle_124 = paddle.multiply(x=x2paddle_122, y=x2paddle_123)
        """

        def gen_name(id):
            return "x" + str(id)

        self.pattern.add_layer(
            "paddle.nn.Sigmoid",
            inputs={"x": "silu-input-0"},
            outputs=[gen_name(0)])
        self.pattern.add_layer(
            "paddle.multiply",
            inputs={"x": "silu-input-0",
                    "y": gen_name(0)},
            outputs=[gen_name(1)])
        self.pattern.build(inputs={"input-0": "silu-input-0", })

    def insert_new_layer(self, graph, parameters, matches):
        new_layer, new_layer_id = self.gen_new_layer(parameters, matches)
        # Determine whether Sigmoid and multiply input are the same
        sigmoid_input = list()
        multiply_input = list()
        for layer_id, layer in matches.items():
            if layer.kernel == "paddle.nn.Sigmoid":
                sigmoid_input.append(layer.inputs['x'])
            if layer.kernel == "paddle.multiply":
                multiply_input.append(layer.inputs['x'])
        if sigmoid_input[0] == multiply_input[0]:
            graph.layers[new_layer_id] = new_layer
            matches.pop(new_layer_id)
        else:
            matches.clear()

    def gen_new_layer(self, parameters, matches):
        layer_id_list = list(matches.keys())
        layer_id_list.sort(key=int)
        layer_inputs = list()
        layer_inputs_ids = list()
        for layer_id, layer in matches.items():
            if layer.kernel == "paddle.nn.Sigmoid":
                layer_inputs.append(layer.inputs)
                layer_inputs_ids.append(layer_id)
            if layer.kernel == "paddle.multiply":
                output_name = layer.outputs[0]
        silu_name = "silu{}".format(self.silu_index)
        self.silu_index += 1
        new_layer = PaddleLayer(
            layer_id_list[0],
            "paddle.nn.Silu",
            inputs=layer_inputs[0],
            outputs=[silu_name, output_name])
        return new_layer, layer_inputs_ids[0]
