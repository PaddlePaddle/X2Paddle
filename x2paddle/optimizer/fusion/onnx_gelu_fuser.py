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


class GeluFuser(FuseBase):
    def __init__(self):
        super(GeluFuser, self).__init__()

    def build_pattern(self):
        """
        code describe:
            x2paddle_332 = paddle.full(dtype='float32', shape=[1], fill_value=1.4142135381698608)
            x2paddle_335 = paddle.full(dtype='float32', shape=[1], fill_value=1.0)
            x2paddle_338 = paddle.full(dtype='float32', shape=[1], fill_value=0.5)
            x2paddle_333 = paddle.divide(x=x2paddle_331, y=x2paddle_332)
            x2paddle_334 = paddle.erf(x=x2paddle_333)
            x2paddle_336 = paddle.add(x=x2paddle_334, y=x2paddle_335)
            x2paddle_337 = paddle.multiply(x=x2paddle_331, y=x2paddle_336)
            x2paddle_339 = paddle.multiply(x=x2paddle_337, y=x2paddle_338)
        """

        def gen_name(id):
            return "x" + str(id)

        self.pattern.add_layer(
            "paddle.full",
            inputs={},
            outputs=[gen_name(0)],
            shape=[1],
            fill_value=1.4142135381698608)
        self.pattern.add_layer(
            "paddle.full",
            inputs={},
            outputs=[gen_name(1)],
            shape=[1],
            fill_value=1.0)
        self.pattern.add_layer(
            "paddle.full",
            inputs={},
            outputs=[gen_name(2)],
            shape=[1],
            fill_value=0.5)
        self.pattern.add_layer(
            "paddle.divide",
            inputs={"x": "gelu-input-0",
                    "y": gen_name(0)},
            outputs=[gen_name(3)])
        self.pattern.add_layer(
            "paddle.erf", inputs={"x": gen_name(3)}, outputs=[gen_name(4)])
        self.pattern.add_layer(
            "paddle.add",
            inputs={"x": gen_name(4),
                    "y": gen_name(1)},
            outputs=[gen_name(5)])
        self.pattern.add_layer(
            "paddle.multiply",
            inputs={"x": "gelu-input-0",
                    "y": gen_name(5)},
            outputs=[gen_name(6)])
        self.pattern.add_layer(
            "paddle.multiply",
            inputs={"x": gen_name(6),
                    "y": gen_name(2)},
            outputs=[gen_name(7)])
        self.pattern.build(inputs={"input-0": "gelu-input-0", })

    def insert_new_layer(self, graph, parameters, matches):
        new_layer, new_layer_id = self.gen_new_layer(parameters, matches)
        graph.layers[new_layer_id] = new_layer
        matches.pop(new_layer_id)

    def gen_new_layer(self, parameters, matches):
        layer_id_list = list(matches.keys())
        layer_id_list.sort(key=int)
        layer_inputs = list()
        layer_inputs_ids = list()
        fill_value_list = list()
        for layer_id, layer in matches.items():
            if layer.kernel == "paddle.divide":
                layer_inputs.append(layer.inputs["x"])
                layer_inputs_ids.append(layer_id)
            if layer.kernel == "paddle.multiply":
                output_name = layer.outputs[0]
        new_layer = PaddleLayer(
            layer_id_list[0],
            "paddle.nn.GELU",
            inputs={"x": layer_inputs[0]},
            outputs=[output_name],
            approximate=False)
        return new_layer, layer_inputs_ids[0]
