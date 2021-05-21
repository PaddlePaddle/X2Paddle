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

import copy
import numpy as np
from x2paddle.optimizer.pattern_matcher import FuseBase
from x2paddle.core.program import PaddleGraph, PaddleLayer
from x2paddle.core.util import *


class InterpolateBilinearFuser(FuseBase):
    def __init__(self):
        super(InterpolateBilinearFuser, self).__init__()
        self.pattenrs = list()

    def build_pattern(self):
        """ 描述需要替换的双线性插值图结构。
        interpolate_bilinear层模式python实现代码示例:
            x2195 = x2181.shape
            x2195 = len(x2195)
            x2196 = x2195 - 2
            x2197 = []
            for _x2199 in range(x2196):
                x2197.append(None)
            x2200 = (x2181, x8, None, None)
            ...
            x2267 = x2266 == 3
            if x2267 :
                raise RaiseException('Exception')
                x2268 = None
            else:
                x2270 = x2181.shape
                x2270 = len(x2270)
                x2271 = x2270 == 4
                if x2271 :
                    x2274 = x2197[0]
                    x2275 = x2197[1]
                    x2233_isinstance = isinstance(x2233, paddle.fluid.Variable)
                    if x2233_isinstance :
                        x2233 = x2233.numpy().tolist()
                    x2276 = paddle.nn.functional.interpolate(x=x2181, size=x2233, scale_factor=x2274, align_corners=False, align_mode=0, mode='bilinear')
                    x2272 = x2276
                else:
                    x2277 = x2181.shape
                    x2277 = len(x2277)
                    x2278 = x2277 == 5
                    if x2278 :
                        raise RaiseException('Exception')
                    else:
                        raise RaiseException('Exception')
                    x2272 = None
                x2268 = x2272
        """

        def gen_name(id):
            return "x" + str(id)

        pattern = PaddleGraph()
        pattern.add_layer(
            "prim.shape",
            inputs={"input": "interpolate-input-0"},
            outputs=[gen_name(9)])
        pattern.add_layer(
            "prim.len", inputs={"input": gen_name(9)}, outputs=[gen_name(9)])
        pattern.add_layer(
            "prim.sub", inputs={"x": gen_name(9)}, outputs=[gen_name(10)], y=2)
        pattern.add_layer("prim.list", inputs={}, outputs=[gen_name(11)])
        pattern.add_layer(
            "prim.loop",
            inputs={"input": gen_name(10)},
            outputs=[gen_name(12.1), gen_name(12.2)])
        loop_layer = pattern.layers[list(pattern.layers.keys())[-1]]
        pattern_block = PaddleGraph(loop_layer)
        pattern_block.add_layer(
            "prim.append",
            inputs={"list": gen_name(11)},
            outputs=[],
            element=None)
        loop_layer.inputs["input-0"] = gen_name(11)
        loop_layer.add_block(pattern_block)
        pattern.add_layer(
            "prim.tuple",
            inputs={
                "input0": "interpolate-input-0",
                "input1": "interpolate-input-4",
            },
            outputs=[gen_name(12)],
            input2=None,
            input3=None)

        pattern.add_layer(
            "prim.eq",
            inputs={"x": "interpolate-input-2"},
            outputs=[gen_name(10.1)],
            y=3)

        pattern.add_layer(
            "prim.if", inputs={"input": gen_name(10.1)},
            outputs=[gen_name(14)])
        if_layer1 = pattern.layers[list(pattern.layers.keys())[-1]]
        pattern_block = PaddleGraph(parent_layer=if_layer1)
        pattern_block.add_layer(
            "prim.exception",
            inputs={},
            outputs=[gen_name(15)],
            input="Exception")
        pattern_block.add_layer(
            "prim.equal", inputs={}, outputs=[gen_name(14)], input=None)
        if_layer1.add_block(pattern_block)
        pattern_block = PaddleGraph(parent_layer=if_layer1)
        pattern_block.add_layer(
            "prim.shape",
            inputs={"input": "interpolate-input-0"},
            outputs=[gen_name(18)])
        pattern_block.add_layer(
            "prim.len", inputs={"input": gen_name(18)}, outputs=[gen_name(18)])
        pattern_block.add_layer(
            "prim.eq", inputs={"x": gen_name(18)}, outputs=[gen_name(19)], y=4)

        pattern_block.add_layer(
            "prim.if", inputs={"input": gen_name(19)}, outputs=[gen_name(20)])
        if_layer2 = pattern_block.layers[list(pattern_block.layers.keys())[-1]]
        pattern_block_block = PaddleGraph(parent_layer=if_layer2)
        pattern_block_block.add_layer(
            "prim.getitem",
            inputs={"list": gen_name(11)},
            outputs=[gen_name(21)],
            element=0)
        pattern_block_block.add_layer(
            "prim.getitem",
            inputs={"list": gen_name(11)},
            outputs=[gen_name(22)],
            element=1)
        pattern_block_block.add_layer(
            "prim.isinstance",
            inputs={"input": "interpolate-input-3"},
            outputs=["interpolate-input-0_isinstance"],
            cls="paddle.fluid.Variable")
        pattern_block_block.add_layer(
            "prim.if", {"input": "interpolate-input-0_isinstance"},
            outputs=["interpolate-input-0_if1"])
        if_layer_isinstance = pattern_block_block.layers[list(
            pattern_block_block.layers.keys())[-1]]
        pattern_block_block_block = PaddleGraph(if_layer_isinstance)
        pattern_block_block_block.add_layer(
            "prim.var2list",
            inputs={"input": "interpolate-input-3"},
            outputs=["interpolate-input-3"])
        if_layer_isinstance.add_block(pattern_block_block_block)
        pattern_block_block_block = PaddleGraph(if_layer_isinstance)
        if_layer_isinstance.add_block(pattern_block_block_block)
        if_layer_isinstance.inputs["input-0"] = "interpolate-input-3"
        pattern_block_block.add_layer(
            "paddle.nn.functional.interpolate",
            inputs={
                "input": "interpolate-input-0",
                "size": "interpolate-input-3",
            },
            outputs=[gen_name(23)])
        pattern_block_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(23)},
            outputs=[gen_name(20)])
        if_layer2.add_block(pattern_block_block)
        pattern_block_block = PaddleGraph(if_layer2)
        pattern_block_block.add_layer(
            "prim.shape",
            inputs={"input": "interpolate-input-0"},
            outputs=[gen_name(24)])
        pattern_block_block.add_layer(
            "prim.len", inputs={"input": gen_name(24)}, outputs=[gen_name(24)])
        pattern_block_block.add_layer(
            "prim.eq", inputs={"x": gen_name(24)}, outputs=[gen_name(25)], y=5)
        pattern_block_block.add_layer(
            "prim.if", inputs={"input": gen_name(25)}, outputs=[gen_name(26)])
        if_layer3 = pattern_block_block.layers[list(
            pattern_block_block.layers.keys())[-1]]
        pattern_block_block_block = PaddleGraph(parent_layer=if_layer3)
        pattern_block_block_block.add_layer(
            "prim.exception",
            inputs={},
            outputs=[gen_name(27)],
            input="Exception")
        if_layer3.add_block(pattern_block_block_block)
        pattern_block_block_block = PaddleGraph(parent_layer=if_layer3)
        pattern_block_block_block.add_layer(
            "prim.exception",
            inputs={},
            outputs=[gen_name(28)],
            input="Exception")
        if_layer3.add_block(pattern_block_block_block)
        pattern_block_block.add_layer(
            "prim.equal", inputs={}, outputs=[gen_name(20)], input=None)
        if_layer2.add_block(pattern_block_block)
        if_layer2.inputs.update({
            "input-0": "interpolate-input-0",
            "input-1": "interpolate-input-3",
            "input-2": "interpolate-input-3",
            "input-3": gen_name(11),
            "input-5": gen_name(11),
        })
        pattern_block.add_layer(
            "prim.equal",
            inputs={"input": gen_name(20)},
            outputs=[gen_name(14)])
        if_layer1.add_block(pattern_block)
        if_layer1.inputs.update({
            'input-2': 'interpolate-input-0',
            'input-4': gen_name(11),
            'input-6': gen_name(11),
            'input-8': 'interpolate-input-0',
            'input-9': 'interpolate-input-3',
            'input-10': 'interpolate-input-0'
        })
        pattern.build(inputs={
            "input-0": "interpolate-input-0",
            "input-1": "interpolate-input-1",
            "input-2": "interpolate-input-2",
            "input-3": "interpolate-input-3",
            "input-4": "interpolate-input-4"
        })
        self.patterns.append(pattern)

    def insert_new_layer(self, graph, parameters, matches):
        new_layer = self.gen_new_layer(parameters, matches)
        global_layers = graph.get_global_layers()
        new_matches = dict()
        is_match = False
        for layer_id, layer in global_layers.items():
            if layer_id == list(matches.keys())[0] and not is_match:
                new_matches[layer_id] = layer
                is_match = True
            if is_match:
                new_matches[layer_id] = layer
                if layer_id == list(matches.keys())[-1]:
                    break
        new_layer_id = new_layer.layer_id
        graph.layers[new_layer_id] = new_layer
        new_matches.pop(new_layer_id)
        matches.clear()
        for layer_id, layer in new_matches.items():
            matches[layer_id] = layer

    def gen_new_layer(self, parameters, matches):
        layers = list()
        layers_id = list(matches.keys())
        layer = matches[layers_id[6]]
        size = layer.inputs["input1"]
        layer = matches[layers_id[19]]
        new_layer = copy.deepcopy(layer)
        layer = matches[layers_id[9]]
        new_layer.outputs[0] = layer.outputs[0]
        new_layer.layer_id = layers_id[7]
        new_layer.inputs["size"] = size
        return new_layer
