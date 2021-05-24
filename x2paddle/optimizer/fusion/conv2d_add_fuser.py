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


class Conv2DAddFuser(FuseBase):
    def __init__(self):
        super(Conv2DAddFuser, self).__init__()
        self.patterns = list()

    def build_pattern(self):
        """ 描述需要替换的conv2d+add图结构。
        conv2d+add层模式python实现代码示例:
        模式一：
        MobilenetV1_Logits_Conv2d_1c_1x1_biases = self.MobilenetV1_Logits_Conv2d_1c_1x1_biases
        conv2d_transpose_14 = paddle.transpose(x=MobilenetV1_Logits_AvgPool_1a_AvgPool, perm=[0, 3, 1, 2])
        MobilenetV1_Logits_Conv2d_1c_1x1_Conv2D = self.conv27(conv2d_transpose_14)
        MobilenetV1_Logits_Conv2d_1c_1x1_Conv2D = paddle.transpose(x=MobilenetV1_Logits_Conv2d_1c_1x1_Conv2D, perm=[0, 2, 3, 1])
        MobilenetV1_Logits_Conv2d_1c_1x1_BiasAdd = paddle.add(x=MobilenetV1_Logits_Conv2d_1c_1x1_Conv2D, y=MobilenetV1_Logits_Conv2d_1c_1x1_biases)
        模式二：
        MobilenetV1_Logits_Conv2d_1c_1x1_biases = self.MobilenetV1_Logits_Conv2d_1c_1x1_biases
        MobilenetV1_Logits_Conv2d_1c_1x1_Conv2D = self.conv27(conv2d_transpose_14)
        MobilenetV1_Logits_Conv2d_1c_1x1_BiasAdd = paddle.add(x=MobilenetV1_Logits_Conv2d_1c_1x1_Conv2D, y=MobilenetV1_Logits_Conv2d_1c_1x1_biases)
        """

        def gen_name(id):
            return "x" + str(id)

        pattern = PaddleGraph()
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(0)])
        pattern.add_layer(
            kernel="paddle.transpose",
            inputs={"x": "conv-input-0"},
            outputs=[gen_name(1)],
            perm=[0, 3, 1, 2])
        pattern.add_layer(
            kernel="paddle.nn.Conv2D",
            inputs={"input": gen_name(1)},
            outputs=[gen_name(2)])
        pattern.add_layer(
            kernel="paddle.transpose",
            inputs={"x": gen_name(2)},
            outputs=[gen_name(2)],
            perm=[0, 2, 3, 1])
        pattern.add_layer(
            kernel="paddle.add",
            inputs={"x": gen_name(2),
                    "y": gen_name(0)},
            outputs=[gen_name(3)])
        pattern.build(inputs={"input-0": "conv-input-0", })
        self.patterns.append(pattern)

        pattern = PaddleGraph()
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(0)])
        pattern.add_layer(
            kernel="paddle.nn.Conv2D",
            inputs={"input": "conv-input-0"},
            outputs=[gen_name(1)])
        pattern.add_layer(
            kernel="paddle.add",
            inputs={"x": gen_name(1),
                    "y": gen_name(0)},
            outputs=[gen_name(2)])
        pattern.build(inputs={"input-0": "conv-input-0", })
        self.patterns.append(pattern)

    def insert_new_layer(self, graph, parameters, matches):
        self.gen_new_layer(matches, graph)
        matches_copy = copy.deepcopy(matches)
        for layer_id, layer in matches_copy.items():
            if layer.kernel not in ["self.create_parameter", "paddle.add"]:
                matches.pop(layer_id)

    def gen_new_layer(self, matches, graph):
        is_transpose = False
        for layer_id, layer in matches.items():
            if layer.kernel == "self.create_parameter":
                bias_name = layer.attrs["attr"]
            if layer.kernel == "paddle.transpose":
                is_transpose = True
            if layer.kernel == "paddle.add":
                output_name = layer.outputs[0]
            if layer.kernel == "paddle.nn.Conv2D":
                conv_id = layer_id
        for layer_id, layer in matches.items():
            if layer.kernel == "paddle.nn.Conv2D":
                layer.attrs["bias_attr"] = bias_name
                if not is_transpose:
                    layer.outputs[1] = output_name
            if layer.kernel == "paddle.transpose":
                if conv_id in graph.edges_in[layer_id]:
                    layer.outputs[0] = output_name
