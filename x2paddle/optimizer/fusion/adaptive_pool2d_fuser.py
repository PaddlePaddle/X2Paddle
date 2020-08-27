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


class AdaptivePool2dFuser(FuseBase):
    def __init__(self):
        super(AdaptivePool2dFuser, self).__init__(graph_type="dygraph")

    def build_pattern(self):
        """ 描述需要替换的adaptive pool2d图结构。
        adaptive pool2d层模式python实现代码示例:
            x72 = [6, 6]
            x73 = x71.shape
            x75 = 'Exception'
            x76 = 9223372036854775807
            x77 = 1
            x78 = len(x73)
            x79 = 2
            x80 = x78 <= x79
            if x80 :
                raise RaiseException(x75)
            x83 = []
            x84 = -2
            x85 = x73[x84: x76: x77]
            x86 = 2
            x87 = len(x85)
            x88 = [x86, x87]
            x89 = min(x88)
            for _x91 in range(x89):
                x92 = x72[_x91]
                x83.append(x92)
            x93 = fluid.layers.adaptive_pool2d(input=x71, pool_size=x83, pool_type='avg')
        """

        def gen_name(id):
            return "x" + str(id)

        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(0)], value=[6, 6])
        self.pattern.add_layer(
            "prim.shape",
            inputs={'input': "pool-input-0"},
            outputs=[gen_name(1)])
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(2)], value=True)
        self.pattern.add_layer(
            "prim.constant",
            inputs={},
            outputs=[gen_name(3)],
            value="Exception")
        self.pattern.add_layer(
            "prim.constant",
            inputs={},
            outputs=[gen_name(4)],
            value=9223372036854775807)
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(5)], value=1)
        self.pattern.add_layer(
            "prim.len", inputs={"input": gen_name(1)}, outputs=[gen_name(6)])
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(7)], value=2)
        self.pattern.add_layer(
            "prim.le",
            inputs={"x": gen_name(6),
                    "y": gen_name(7)},
            outputs=[gen_name(8)])
        self.pattern.add_layer("prim.if", {'input': gen_name(8)}, [gen_name(9)])
        if_layer = self.pattern.layers[list(self.pattern.layers.keys())[-1]]
        pattern_block0 = PaddleGraph(if_layer, graph_type="dygraph")
        pattern_block0.add_layer(
            "prim.exception",
            inputs={"input": gen_name(3)},
            outputs=[gen_name(9)])
        if_layer.inputs["input-0"] = gen_name(3)
        if_layer.add_block(pattern_block0)
        pattern_block1 = PaddleGraph(if_layer, graph_type="dygraph")
        if_layer.add_block(pattern_block1)
        self.pattern.add_layer("prim.list", inputs={}, outputs=[gen_name(10)])
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(11)], value=-2)
        self.pattern.add_layer(
            "prim.slice",
            inputs={
                "input": gen_name(1),
                "start": gen_name(11),
                "end": gen_name(4),
                "step": gen_name(5)
            },
            outputs=[gen_name(12)])
        self.pattern.add_layer(
            "prim.constant", inputs={}, outputs=[gen_name(13)], value=2)
        self.pattern.add_layer(
            "prim.len", inputs={"input": gen_name(12)}, outputs=[gen_name(14)])
        self.pattern.add_layer(
            "prim.list",
            inputs={"input0": gen_name(13),
                    "input1": gen_name(14)},
            outputs=[gen_name(15)])
        self.pattern.add_layer(
            "prim.min", inputs={"input": gen_name(15)}, outputs=[gen_name(16)])
        self.pattern.add_layer("prim.loop", {'input': gen_name(16)},
                               [gen_name(17), gen_name(18)])
        loop_layer = self.pattern.layers[list(self.pattern.layers.keys())[-1]]
        pattern_block = PaddleGraph(loop_layer, graph_type="dygraph")
        pattern_block.add_layer(
            "prim.getitem",
            inputs={"list": gen_name(0),
                    "index": gen_name(18)},
            outputs=[gen_name(19)])
        pattern_block.add_layer(
            "prim.append",
            inputs={"list": gen_name(10),
                    "index": gen_name(19)},
            outputs=[gen_name(20)])
        loop_layer.inputs["input-0"] = gen_name(0)
        loop_layer.inputs["input-2"] = gen_name(10)
        loop_layer.add_block(pattern_block)
        pool_attrs = {'pool_type': string("avg")}
        self.pattern.add_layer(
            "fluid.layers.adaptive_pool2d",
            inputs={'input': "pool-input-0",
                    "pool_size": gen_name(10)},
            outputs=[gen_name(21)],
            **pool_attrs)
        self.pattern.build(inputs={"input-0": "pool-input-0"})

    def insert_new_layer(self, graph, parameters, matches):
        parameters = graph.parameters
        new_layer = self.gen_new_layer(parameters, matches)
        new_layer_id = list(matches.keys())[0]
        graph.layers[new_layer_id] = new_layer
        matches.pop(new_layer_id)

    def gen_new_layer(self, parameters, matches):
        layers_id = list(matches.keys())
        layer = matches[layers_id[0]]
        pool_size = layer.attrs["value"]
        layer = matches[layers_id[1]]
        input_name = layer.inputs["input"]
        layer = matches[layers_id[-1]]
        output_name = layer.outputs[0]
        pool_type = layer.attrs["pool_type"]
        attrs = dict()
        attrs["pool_size"] = pool_size
        attrs["pool_type"] = pool_type
        new_layer = PaddleLayer(
            layers_id[0],
            "fluid.layers.adaptive_pool2d",
            inputs={"input": input_name},
            outputs=[output_name],
            **attrs)
        return new_layer
