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


class DropoutFuser(FuseBase):
    def __init__(self):
        super(DropoutFuser, self).__init__()

    def build_pattern(self):
        """ 描述需要替换的constant图结构。
        constant层模式python实现代码示例:
            x3 = 10
            for _x70 in range(x3):
                ...
        """
        self.pattern.add_layer(
            "paddle.nn.Dropout",
            inputs={"input": "dropout-input-0"},
            outputs=["dropout0", "x1"])
        self.pattern.build(inputs={"input-0": "dropout-input-0"})
        self.pattern.outputs = ["dropout0", "x1"]

    def insert_new_layer(self, graph, parameters, matches):
        def replace_value(layer_connect, match_name, match_input):
            for k, v in layer_connect.inputs.items():
                if v == match_name:
                    layer_connect.inputs[k] = match_input
                    break
            if layer_connect.kernel == "prim.loop" or \
            layer_connect.kernel == "prim.if":
                for block in layer_connect.blocks:
                    for b_layer_id, b_layer in block.layers.items():
                        if block.edges_in.get(b_layer_id, 0) != 0 and  \
                        -1 in block.edges_in[b_layer_id]:
                            replace_value(b_layer, match_name, match_input)

        layer_id = list(matches.keys())[0]
        layer = list(matches.values())[0]
        layer_output_name = layer.outputs[1]
        layer_input = layer.inputs["input"]
        if graph.edges_out.get(layer_id, 0) != 0:
            for layer_id_out in graph.edges_out[layer_id]:
                layer_connect = graph.layers[layer_id_out]
                replace_value(layer_connect, layer_output_name, layer_input)
