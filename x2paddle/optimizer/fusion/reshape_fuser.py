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

import numpy as np
from x2paddle.optimizer.pattern_matcher import FuseBase
from x2paddle.core.program import PaddleGraph, PaddleLayer
from x2paddle.core.util import *


class ReshapeFuser(FuseBase):
    def __init__(self):
        super(ReshapeFuser, self).__init__()

    def build_pattern(self):
        """ 描述需要替换的reshape图结构。
        reshape层模式python实现代码示例:
            x165 = int(x164)
            x166 = [x158, x159, x165]
            x167 = paddle.reshape(x=x157, shape=x166)
        """

        def gen_name(id):
            return "x" + str(id)

        self.pattern.add_layer(
            "prim.int",
            inputs={"input": "reshape-input-0"},
            outputs=[gen_name(0)])
        self.pattern.add_layer(
            "prim.list",
            inputs={
                "input0": "reshape-input-1",
                "input1": "reshape-input-2",
                "input2": gen_name(0)
            },
            outputs=[gen_name(1)])
        self.pattern.add_layer(
            "paddle.reshape",
            inputs={"x": "reshape-input-3",
                    "shape": gen_name(1)},
            outputs=[gen_name(2)])
        self.pattern.build(inputs={
            "input-0": "reshape-input-0",
            "input-1": "reshape-input-1",
            "input-2": "reshape-input-2",
            "input-3": "reshape-input-3",
        })

    def insert_new_layer(self, graph, parameters, matches):
        self.update_layer(matches)
        matches.pop(list(matches.keys())[1])
        matches.pop(list(matches.keys())[1])

    def update_layer(self, matches):
        layers_id = list(matches.keys())
        layer = matches[layers_id[0]]
        int_input_name = layer.inputs["input"]
        output_name = layer.outputs[0]
        layer = matches[layers_id[1]]
        for key, input_name in layer.inputs.items():
            if input_name == output_name:
                layer.inputs[key] = int_input_name
