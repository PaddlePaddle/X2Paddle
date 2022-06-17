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


class Div2Scale(FuseBase):
    def __init__(self):
        super(Div2Scale, self).__init__()

    def build_pattern(self):
        """
        code describe:
            x2paddle_296 = paddle.full(dtype='float32', shape=[1], fill_value=8.0)
            x2paddle_293 = paddle.transpose(x=x2paddle_292, perm=[0, 2, 1, 3])
            x2paddle_294 = paddle.transpose(x=x2paddle_260, perm=[0, 2, 3, 1])
            x2paddle_295 = paddle.matmul(x=x2paddle_293, y=x2paddle_294)
            x2paddle_297 = paddle.divide(x=x2paddle_295, y=x2paddle_296)
        """

        def gen_name(id):
            return "x" + str(id)

        self.pattern.add_layer(
            "paddle.full",
            inputs={},
            outputs=[gen_name(0)],
            shape=[1],
            fill_value=8)
        self.pattern.add_layer(
            "paddle.transpose",
            inputs={"x": "div2scale-input-0"},
            outputs=[gen_name(1)],
            perm=[0, 2, 1, 3])
        self.pattern.add_layer(
            "paddle.transpose",
            inputs={"x": "div2scale-input-1"},
            outputs=[gen_name(2)],
            perm=[0, 2, 1, 3])
        self.pattern.add_layer(
            "paddle.matmul",
            inputs={"x": gen_name(1),
                    "y": gen_name(2)},
            outputs=[gen_name(3)])
        self.pattern.add_layer(
            "paddle.divide",
            inputs={"x": gen_name(3),
                    "y": gen_name(0)},
            outputs=[gen_name(4)])
        self.pattern.build(inputs={
            "input-0": "div2scale-input-0",
            "input-1": "div2scale-input-1",
        })

    def insert_new_layer(self, graph, parameters, matches):
        new_layer, new_layer_id = self.gen_new_layer(parameters, matches)
        graph.layers[new_layer_id] = new_layer
        matches_copy = copy.deepcopy(matches)
        for layer_id, layer in matches_copy.items():
            if layer.kernel in ["paddle.transpose", "paddle.matmul"]:
                matches.pop(layer_id)
        matches.pop(new_layer_id)

    def gen_new_layer(self, parameters, matches):
        layer_id_list = list(matches.keys())
        layer_id_list.sort(key=int)
        layer_inputs = list()
        layer_inputs_ids = list()
        fill_value = 0
        for layer_id, layer in matches.items():
            if layer.kernel == "paddle.full":
                fill_value = layer.attrs["fill_value"]
            if layer.kernel == "paddle.divide":
                layer_inputs.append(layer.inputs["x"])
                layer_inputs_ids.append(layer_id)
                output_name = layer.outputs[0]
        new_layer = PaddleLayer(
            layer_id_list[0],
            "paddle.scale",
            inputs={"x": layer_inputs[0]},
            outputs=[output_name],
            scale=1 / fill_value)
        return new_layer, layer_inputs_ids[0]
