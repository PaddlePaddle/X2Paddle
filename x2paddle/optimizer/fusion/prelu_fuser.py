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
from collections import OrderedDict
from x2paddle.optimizer.pattern_matcher import FuseBase
from x2paddle.core.program import PaddleGraph, PaddleLayer
from x2paddle.core.util import *


class PReLUFuser(FuseBase):
    def __init__(self):
        self.prelu_index = 0
        super(PReLUFuser, self).__init__()

    def build_pattern(self):
        """ 描述需要替换的prelu图结构。
        prelu层模式python实现代码示例:
            conv2_alphas = self.conv2_alphas
            conv2_mul_1_y = paddle.full(dtype='float32', shape=[1], fill_value=0.5)
            conv2_Relu = self.relu1(conv2_Conv2D)
            conv2_Abs = paddle.abs(x=conv2_Conv2D)
            conv2_sub = paddle.subtract(x=conv2_Conv2D, y=conv2_Abs)
            conv2_mul = paddle.multiply(x=conv2_alphas, y=conv2_sub, axis=1)
            conv2_mul_1 = paddle.multiply(x=conv2_mul, y=conv2_mul_1_y, axis=1)
            conv2_add = paddle.add(x=conv2_Relu, y=conv2_mul_1)
        """

        def gen_name(id):
            return "x" + str(id)

        self.pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(0)])
        self.pattern.add_layer(
            "paddle.full",
            inputs={},
            outputs=[gen_name(1)],
            shape=[1],
            fill_value=0.5)
        self.pattern.add_layer(
            "paddle.nn.ReLU",
            inputs={"x": "prelu-input-0"},
            outputs=[gen_name(2)])
        self.pattern.add_layer(
            "paddle.abs", inputs={"x": "prelu-input-0"}, outputs=[gen_name(3)])
        self.pattern.add_layer(
            "paddle.subtract",
            inputs={"x": "prelu-input-0",
                    "y": gen_name(3)},
            outputs=[gen_name(4)])
        self.pattern.add_layer(
            "paddle.multiply",
            inputs={"x": gen_name(0),
                    "y": gen_name(4)},
            outputs=[gen_name(5)])
        self.pattern.add_layer(
            "paddle.multiply",
            inputs={"x": gen_name(5),
                    "y": gen_name(1)},
            outputs=[gen_name(6)])
        self.pattern.add_layer(
            "paddle.add",
            inputs={"x": gen_name(2),
                    "y": gen_name(6)},
            outputs=[gen_name(7)])
        self.pattern.build(inputs={"input-0": "prelu-input-0", })

    def insert_new_layer(self, graph, parameters, matches):
        new_layers, last_layer_id = self.gen_new_layer(matches, parameters,
                                                       graph)
        matches_copy = copy.deepcopy(matches)
        for layer_id, layer in matches_copy.items():
            for i in range(3):
                if layer_id == new_layers[i].id:
                    matches.pop(new_layers[i].id)
        prefix_layers = OrderedDict()
        mid_layers = OrderedDict()
        suffix_layers = OrderedDict()
        is_need_id = False
        for layer_id, layer in graph.layers.items():
            if is_need_id:
                suffix_layers[layer_id] = layer
            else:
                if layer_id == last_layer_id:
                    for i in range(3):
                        mid_layers[new_layers[i].id] = new_layers[i]
                    is_need_id = True
                prefix_layers[layer_id] = layer
        prefix_layers.update(mid_layers)
        prefix_layers.update(suffix_layers)
        graph.layers = prefix_layers

    def gen_new_layer(self, matches, parameters, graph):
        layer_id_list = list(matches.keys())
        layer_id_list.sort(key=int)
        for layer_id, layer in matches.items():
            if layer.kernel == "paddle.nn.ReLU":
                input_name = layer.inputs["x"]
            if layer.kernel == "self.create_parameter":
                param_name = layer.outputs[0]
            if layer.kernel == "paddle.add":
                output_name = layer.outputs[0]
        transpose0 = PaddleLayer(
            id=layer_id_list[-1] + "_1",
            kernel="paddle.transpose",
            inputs={"x": input_name},
            outputs=["{}_transpose_for_prelu".format(input_name)],
            perm=[0, 3, 1, 2])
        prelu_name = "merge_prelu{}".format(self.prelu_index)
        self.prelu_index += 1
        param = parameters[param_name]
        c = param.shape[0]
        prelu = PaddleLayer(
            id=layer_id_list[-1] + "_2",
            kernel="paddle.nn.PReLU",
            inputs={"input": "{}_transpose_for_prelu".format(input_name)},
            outputs=[prelu_name, "{}_prelu".format(input_name)],
            num_parameters=c,
            weight_attr=string(param_name))
        transpose1 = PaddleLayer(
            id=layer_id_list[-1] + "_3",
            kernel="paddle.transpose",
            inputs={"x": "{}_prelu".format(input_name)},
            outputs=[output_name],
            perm=[0, 2, 3, 1])
        return [transpose0, prelu, transpose1], layer_id_list[-1]
