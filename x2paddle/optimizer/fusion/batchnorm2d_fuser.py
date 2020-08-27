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
        super(BatchNorm2dFuser, self).__init__(graph_type="dygraph")

    def build_pattern(self):
        """ 描述需要替换的batchnorm2d图结构。
        batchnorm2d层模式python实现代码示例:
            x2209 = 1
            x2212 = 'Exception'
            x2213 = 4
            x2214 = x2207.shape
            x2214 = len(x2214)
            x2215 = x2214 != x2213
            if x2215 :
                raise RaiseException(x2212)
            x2218 = False
            if x2218 :
                x2220 = self.x2220
                x2221 = x2220 + x2209
                self.x2220 = x2221
            x2227 = False
            if x2227 :
                x2230 = x2207.shape
                x2231 = 'Exception'
                x2233 = 0
                x2234 = 2
                x2235 = 1
                x2236 = x2230[x2233]
                x2237 = len(x2230)
                x2238 = x2237 - x2234
                x2241 = x2236
                for _x2240 in range(x2238):
                    x2242 = _x2240 + x2234
                    x2243 = x2230[x2242]
                    x2244 = x2241 * x2243
                    x2239 = x2244
                x2245 = x2239 == x2235
                if x2245 :
                    raise RaiseException(x2231)
            x2248 = self.batchnorm41(x2207)
        """

        def gen_name(id):
            return "x" + str(id)

        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(0)], value=1)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(1)], value=0.1)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(2)], value=0.001)
        self.pattern.add_layer(
            "prim.constant",
            inputs={},
            outputs=[gen_name(3)],
            value="Exception")
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(4)], value=4)
        self.pattern.add_layer(
            "prim.shape", inputs={'input': "bn-input-0"},
            outputs=[gen_name(5)])
        self.pattern.add_layer(
            "prim.len", inputs={'input': gen_name(5)}, outputs=[gen_name(5)])
        self.pattern.add_layer(
            "prim.ne",
            inputs={"x": gen_name(5),
                    "y": gen_name(4)},
            outputs=[gen_name(6)])
        self.pattern.add_layer("prim.if", {'input': gen_name(6)}, [gen_name(7)])
        if_layer1 = self.pattern.layers[list(self.pattern.layers.keys())[-1]]
        pattern_block0 = PaddleGraph(if_layer1, graph_type="dygraph")
        pattern_block0.add_layer(
            "prim.exception",
            inputs={"input": gen_name(3)},
            outputs=[gen_name(8)])
        if_layer1.inputs["input-0"] = gen_name(3)
        if_layer1.add_block(pattern_block0)
        pattern_block1 = PaddleGraph(if_layer1, graph_type="dygraph")
        if_layer1.add_block(pattern_block1)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(9)], value=False)
        self.pattern.add_layer("prim.if", {'input': gen_name(9)},
                               [gen_name(10)])
        if_layer2 = self.pattern.layers[list(self.pattern.layers.keys())[-1]]
        pattern_block0 = PaddleGraph(if_layer2, graph_type="dygraph")
        pattern_block0.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[gen_name(11)],
            value="params[{}]".format(string(gen_name(11))))
        pattern_block0.add_layer(
            "prim.add",
            inputs={"x": gen_name(11),
                    "y": gen_name(0)},
            outputs=[gen_name(12)])
        pattern_block0.add_layer(
            "prim.set_attr",
            inputs={"input": gen_name(12)},
            outputs=["self." + gen_name(11)])
        if_layer2.inputs["input-0"] = gen_name(0)
        if_layer2.add_block(pattern_block0)
        pattern_block1 = PaddleGraph(if_layer2, graph_type="dygraph")
        if_layer2.add_block(pattern_block1)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(13)], value=True)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(14)], value=False)
        self.pattern.add_layer("prim.if", {'input': gen_name(14)},
                               [gen_name(15)])
        if_layer3 = self.pattern.layers[list(self.pattern.layers.keys())[-1]]
        pattern_block0 = PaddleGraph(if_layer3, graph_type="dygraph")
        pattern_block0.add_layer(
            "prim.shape",
            inputs={'input': "bn-input-0"},
            outputs=[gen_name(16)])
        pattern_block0.add_layer(
            "prim.constant",
            inputs={},
            outputs=[gen_name(17)],
            value="Exception")
        pattern_block0.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(18)], value=True)
        pattern_block0.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(19)], value=0)
        pattern_block0.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(20)], value=2)
        pattern_block0.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(21)], value=1)
        pattern_block0.add_layer(
            "prim.getitem",
            inputs={"list": gen_name(16),
                    "index": gen_name(19)},
            outputs=[gen_name(22)])
        pattern_block0.add_layer(
            "prim.len", inputs={"input": gen_name(16)}, outputs=[gen_name(23)])
        pattern_block0.add_layer(
            "prim.sub",
            inputs={"x": gen_name(23),
                    "y": gen_name(20)},
            outputs=[gen_name(24)])
        pattern_block0.add_layer(
            "prim.equal",
            inputs={"input": gen_name(22)},
            outputs=[gen_name(25)])
        pattern_block0.add_layer(
            "prim.loop",
            inputs={"input": gen_name(24)},
            outputs=[gen_name(26), gen_name(27)])
        loop_layer = pattern_block0.layers[list(pattern_block0.layers.keys())[
            -1]]
        pattern_block0_block0 = PaddleGraph(loop_layer, graph_type="dygraph")
        pattern_block0_block0.add_layer(
            "prim.add",
            inputs={"x": gen_name(27),
                    "y": gen_name(20)},
            outputs=[gen_name(28)])
        pattern_block0_block0.add_layer(
            "prim.getitem",
            inputs={"list": gen_name(16),
                    "index": gen_name(28)},
            outputs=[gen_name(29)])
        pattern_block0_block0.add_layer(
            "prim.mul",
            inputs={"x": gen_name(25),
                    "y": gen_name(29)},
            outputs=[gen_name(30)])
        pattern_block0_block0.add_layer(
            "prim.equal",
            inputs={"input": gen_name(30)},
            outputs=[gen_name(26)])
        loop_layer.inputs["input-1"] = gen_name(20)
        loop_layer.inputs["input-2"] = gen_name(16)
        loop_layer.inputs["input-3"] = gen_name(25)
        loop_layer.add_block(pattern_block0_block0)
        pattern_block0.add_layer(
            "prim.eq",
            inputs={"x": gen_name(26),
                    "y": gen_name(21)},
            outputs=[gen_name(31)])
        pattern_block0.add_layer(
            "prim.if", inputs={"input": gen_name(31)}, outputs=[gen_name(32)])
        if_layer31 = pattern_block0.layers[list(pattern_block0.layers.keys())[
            -1]]
        pattern_block0_block0 = PaddleGraph(if_layer31, graph_type="dygraph")
        pattern_block0_block0.add_layer(
            "prim.exception",
            inputs={"input": gen_name(17)},
            outputs=[gen_name(33)])
        if_layer31.inputs["input-0"] = gen_name(17)
        if_layer31.add_block(pattern_block0_block0)
        pattern_block0_block1 = PaddleGraph(if_layer31, graph_type="dygraph")
        if_layer31.add_block(pattern_block0_block1)
        if_layer3.inputs["input-0"] = "bn-input-0"
        if_layer3.add_block(pattern_block0)
        pattern_block1 = PaddleGraph(if_layer3, graph_type="dygraph")
        if_layer3.add_block(pattern_block1)
        self.pattern.add_layer(
            "fluid.dygraph.BatchNorm",
            inputs={"input": "bn-input-0"},
            outputs=[gen_name(34), gen_name(35)],
            is_test=True,
            num_channels=160,
            momentum=0.1,
            epsilon=0.001)
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
