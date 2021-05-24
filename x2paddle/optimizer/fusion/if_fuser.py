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


class IfFuser(FuseBase):
    def __init__(self):
        super(IfFuser, self).__init__()

    def build_pattern(self):
        """ 描述需要替换的if图结构。
        if层模式python实现代码示例:
            x81 = 'relu' in {'layer4': 'out', 'layer3': 'aux'}
            if x81 :
                ...
        """
        self.pattern.add_layer(
            "prim.if", inputs={"input": "if-input-0"}, outputs=["x0"])
        self.pattern.build(inputs={"input-0": "if-input-0"})

    def insert_new_layer(self, graph, parameters, matches):
        layer_id = list(matches.keys())[0]
        layer = list(matches.values())[0]
        if "input" not in layer.inputs:
            matches.pop(layer_id)
            return
        for id in graph.edges_in[layer_id]:
            input_layer = graph.layers[id]
            input_layer_id = id
            if input_layer.outputs == [layer.inputs["input"]]:
                if input_layer.kernel == "prim.if":
                    matches.pop(layer_id)
                    return
                input_id = id
                break
        if list(layer.inputs.values()).count(input_layer.outputs[0]) > 1 or \
               (input_layer_id in graph.edges_out and len(graph.edges_out[input_layer_id]) > 1):
            matches.pop(layer_id)
            return
        func_name = input_layer.kernel.replace(".", "_")
        if func_name in ["prim_if", "prim_loop"]:
            matches.pop(layer_id)
            return
        from x2paddle.op_mapper.pytorch2paddle import prim2code
        func = getattr(prim2code, func_name)
        line = func(input_layer, is_return_line=True)
        layer.attrs["input"] = line
        layer.inputs.pop("input")
        matches.pop(layer_id)
        if len(input_layer.outputs) == 1:
            matches[input_id] = input_layer
