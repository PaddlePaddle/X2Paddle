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


class BatchNorm2dFuser(FuseBase):
    def __init__(self):
        super(BatchNorm2dFuser, self).__init__()

    def build_pattern(self):
        """ 描述需要替换的batchnorm2d图结构。
        batchnorm2d层模式python实现代码示例:
            x2011 = x2009.shape
            x2011 = len(x2011)
            x2012 = x2011 != 4
            if x2012 :
                raise RaiseException('Exception')
            if False :
                x2026 = x2009.shape
                x2027 = x2026[0]
                x2028 = len(x2026)
                x2029 = x2028 - 2
                x2032 = x2027
                for _x2031 in range(x2029):
                    x2033 = _x2031 + 2
                    x2034 = x2026[x2033]
                    x2035 = x2032 * x2034
                    x2030 = x2035
                x2036 = x2030 == 1
                if x2036 :
                    raise RaiseException('Exception')
            x2039 = self.batchnorm50(x2009)
        """

        def gen_name(id):
            return "x" + str(id)

        self.pattern.add_layer(
            "prim.shape", inputs={'input': "bn-input-0"},
            outputs=[gen_name(0)])
        self.pattern.add_layer(
            "prim.len", inputs={'input': gen_name(0)}, outputs=[gen_name(0)])
        self.pattern.add_layer(
            "prim.ne", inputs={"x": gen_name(0)}, outputs=[gen_name(1)], y=4)
        self.pattern.add_layer("prim.if", {'input': gen_name(1)}, [gen_name(2)])
        if_layer1 = self.pattern.layers[list(self.pattern.layers.keys())[-1]]
        pattern_block0 = PaddleGraph(parent_layer=if_layer1)
        pattern_block0.add_layer(
            "prim.exception",
            inputs={},
            outputs=[gen_name(3)],
            input="Exception")
        if_layer1.add_block(pattern_block0)
        pattern_block1 = PaddleGraph(parent_layer=if_layer1)
        if_layer1.add_block(pattern_block1)
        self.pattern.add_layer("prim.if", {}, [gen_name(4)], input=False)
        if_layer2 = self.pattern.layers[list(self.pattern.layers.keys())[-1]]
        pattern_block0 = PaddleGraph(parent_layer=if_layer2)
        pattern_block0.add_layer(
            "prim.shape", inputs={'input': "bn-input-0"},
            outputs=[gen_name(5)])
        pattern_block0.add_layer(
            "prim.getitem",
            inputs={"list": gen_name(5)},
            outputs=[gen_name(6)],
            index=0)
        pattern_block0.add_layer(
            "prim.len", inputs={"input": gen_name(5)}, outputs=[gen_name(7)])
        pattern_block0.add_layer(
            "prim.sub", inputs={"x": gen_name(7)}, outputs=[gen_name(8)], y=2)
        pattern_block0.add_layer(
            "prim.equal", inputs={"input": gen_name(6)}, outputs=[gen_name(9)])
        pattern_block0.add_layer(
            "prim.loop",
            inputs={"input": gen_name(8)},
            outputs=[gen_name(8.1), gen_name(10)])
        loop_layer = pattern_block0.layers[list(pattern_block0.layers.keys())[
            -1]]
        pattern_block0_block0 = PaddleGraph(parent_layer=loop_layer)
        pattern_block0_block0.add_layer(
            "prim.add", inputs={"x": gen_name(10)}, outputs=[gen_name(11)], y=2)
        pattern_block0_block0.add_layer(
            "prim.getitem",
            inputs={"list": gen_name(5),
                    "index": gen_name(11)},
            outputs=[gen_name(12)])
        pattern_block0_block0.add_layer(
            "prim.mul",
            inputs={"x": gen_name(9),
                    "y": gen_name(12)},
            outputs=[gen_name(13)])
        pattern_block0_block0.add_layer(
            "prim.equal",
            inputs={"input": gen_name(13)},
            outputs=[gen_name(8.1)])
        loop_layer.inputs["input-1"] = gen_name(5)
        loop_layer.inputs["input-2"] = gen_name(9)
        loop_layer.add_block(pattern_block0_block0)
        pattern_block0.add_layer(
            "prim.eq", inputs={"x": gen_name(8.1)}, outputs=[gen_name(14)], y=1)
        pattern_block0.add_layer(
            "prim.if", inputs={"input": gen_name(14)}, outputs=[gen_name(15)])
        if_layer21 = pattern_block0.layers[list(pattern_block0.layers.keys())[
            -1]]
        pattern_block0_block0 = PaddleGraph(parent_layer=if_layer21)
        pattern_block0_block0.add_layer(
            "prim.exception",
            inputs={},
            outputs=[gen_name(15)],
            input="Exception")
        if_layer21.add_block(pattern_block0_block0)
        pattern_block0_block1 = PaddleGraph(parent_layer=if_layer21)
        if_layer21.add_block(pattern_block0_block1)
        if_layer2.add_block(pattern_block0)
        pattern_block1 = PaddleGraph(parent_layer=if_layer2)
        if_layer2.add_block(pattern_block1)
        if_layer2.inputs["input-0"] = "bn-input-0"
        self.pattern.add_layer(
            "paddle.nn.BatchNorm",
            inputs={"input": "bn-input-0"},
            outputs=[gen_name(16), gen_name(17)],
            is_test=True)
        self.pattern.build(inputs={"input-0": "bn-input-0"})

    def insert_new_layer(self, graph, parameters, matches):
        new_layer = self.gen_new_layer(parameters, matches)
        new_layer_id = list(matches.keys())[0]
        graph.layers[new_layer_id] = new_layer
        matches.pop(new_layer_id)

    def gen_new_layer(self, parameters, matches):
        layers_id = list(matches.keys())
        layer = matches[layers_id[-1]]
        return layer
