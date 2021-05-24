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
import copy
from x2paddle.optimizer.pattern_matcher import FuseBase
from x2paddle.core.program import PaddleGraph, PaddleLayer
from x2paddle.core.util import *


class AdaptivePool2dFuser(FuseBase):
    def __init__(self):
        super(AdaptivePool2dFuser, self).__init__()
        self.patterns = list()

    def build_pattern(self):
        """ 描述需要替换的adaptive pool2d图结构。
        adaptive pool2d层模式python实现代码示例:
            模式一：
            x68 = prim.shape(input=x60)
            x69 = len(x68)
            x70 = x69 <= 2
            if x70 :
                raise RaiseException('Exception')
            x73 = []
            x74 = x68[-2: 2147483647: 1]
            x75 = len(x74)
            x76 = [2, x75]
            x77 = min(x76)
            for _x79 in range(x77):
                x80 = [6, 6][_x79]
                x73.append(x80)
            x81 = paddle.nn.functional.adaptive_avg_pool2d(input=x60, pool_size=x73, pool_type='avg')

            模式二：
            x64 = x60.shape
            x65 = len(x64)
            x66 = x65 > 2
            if x66 :
                pass
            else:
                raise RaiseException('AssertionError: ')
            x69 = self.pool2d3(x60)
        """

        def gen_name(id):
            return "x" + str(id)

        # 模式一：
        pattern = PaddleGraph()
        pattern.add_layer(
            "prim.shape",
            inputs={'input': "pool-input-0"},
            outputs=[gen_name(1)])
        pattern.add_layer(
            "prim.len", inputs={"input": gen_name(1)}, outputs=[gen_name(6)])
        pattern.add_layer(
            "prim.le", inputs={"x": gen_name(6)}, outputs=[gen_name(8)], y=2)
        pattern.add_layer("prim.if", {'input': gen_name(8)}, [gen_name(9)])
        if_layer = pattern.layers[list(pattern.layers.keys())[-1]]
        pattern_block0 = PaddleGraph(parent_layer=if_layer)
        pattern_block0.add_layer(
            "prim.exception",
            inputs={},
            outputs=[gen_name(9)],
            input="Exception")
        if_layer.add_block(pattern_block0)
        pattern_block1 = PaddleGraph(parent_layer=if_layer)
        if_layer.add_block(pattern_block1)
        pattern.add_layer("prim.list", inputs={}, outputs=[gen_name(10)])
        pattern.add_layer(
            "prim.slice",
            inputs={"input": gen_name(1), },
            outputs=[gen_name(12)])
        pattern.add_layer(
            "prim.len", inputs={"input": gen_name(12)}, outputs=[gen_name(14)])
        pattern.add_layer(
            "prim.list",
            inputs={"input1": gen_name(14)},
            outputs=[gen_name(15)])
        pattern.add_layer(
            "prim.min", inputs={"input": gen_name(15)}, outputs=[gen_name(16)])
        pattern.add_layer("prim.loop", {'input': gen_name(16)},
                          [gen_name(17), gen_name(18)])
        loop_layer = pattern.layers[list(pattern.layers.keys())[-1]]
        pattern_block = PaddleGraph(loop_layer)
        pattern_block.add_layer(
            "prim.getitem",
            inputs={"index": gen_name(18)},
            outputs=[gen_name(19)])
        pattern_block.add_layer(
            "prim.append",
            inputs={"list": gen_name(10),
                    "index": gen_name(19)},
            outputs=[gen_name(20)])
        loop_layer.inputs["input-0"] = gen_name(10)
        loop_layer.add_block(pattern_block)
        pool_attrs = {'pool_type': string("avg")}
        pattern.add_layer(
            "paddle.nn.functional.adaptive_avg_pool2d",
            inputs={'input': "pool-input-0",
                    "pool_size": gen_name(10)},
            outputs=[gen_name(21)],
            **pool_attrs)
        pattern.build(inputs={"input-0": "pool-input-0", })
        self.patterns.append(pattern)

        # 模式二：
        pattern = PaddleGraph()
        pattern.add_layer(
            "prim.shape",
            inputs={'input': "pool-input-0"},
            outputs=[gen_name(0)])
        pattern.add_layer(
            "prim.len", inputs={"input": gen_name(0)}, outputs=[gen_name(1)])
        pattern.add_layer(
            "prim.gt", inputs={"x": gen_name(1)}, outputs=[gen_name(2)], y=2)
        pattern.add_layer("prim.if", {'input': gen_name(2)}, [gen_name(3)])
        if_layer = pattern.layers[list(pattern.layers.keys())[-1]]
        pattern_block0 = PaddleGraph(parent_layer=if_layer)
        if_layer.add_block(pattern_block0)
        pattern_block1 = PaddleGraph(parent_layer=if_layer)
        pattern_block1.add_layer(
            "prim.exception",
            inputs={},
            outputs=[gen_name(4)],
            input="Exception")
        if_layer.add_block(pattern_block1)
        pattern.add_layer(
            "paddle.nn.AdaptiveAvgPool2D",
            inputs={"input": "pool-input-0"},
            outputs=["pool1", gen_name(5)])
        pattern.build(inputs={
            "input-0": "pool-input-0",
            "input-1": "pool-input-0",
        })
        self.patterns.append(pattern)

    def insert_new_layer(self, graph, parameters, matches):
        parameters = graph.parameters
        new_layer = self.gen_new_layer(parameters, matches)
        new_layer_id = list(matches.keys())[0]
        graph.layers[new_layer_id] = new_layer
        matches.pop(new_layer_id)

    def gen_new_layer(self, parameters, matches):
        layers_id = list(matches.keys())
        if matches[layers_id[
                -1]].kernel == "paddle.nn.functional.adaptive_avg_pool2d":
            layer = matches[layers_id[11]]
            pool_size = layer.attrs["list"]
            layer = matches[layers_id[0]]
            input_name = layer.inputs["input"]
            layer = matches[layers_id[-1]]
            output_name = layer.outputs[0]
            attrs = dict()
            attrs["output_size"] = pool_size
            new_layer = PaddleLayer(
                layers_id[0],
                "paddle.nn.functional.adaptive_avg_pool2d",
                inputs={"x": input_name},
                outputs=[output_name],
                **attrs)
        else:
            new_layer = copy.deepcopy(matches[layers_id[-1]])
        return new_layer
