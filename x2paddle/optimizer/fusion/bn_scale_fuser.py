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


class BNScaleFuser(FuseBase):
    def __init__(self):
        super(BNScaleFuser, self).__init__()
        patterns = list()

    def build_pattern(self):
        """ 描述需要替换的batchnorm2d图结构。
        batchnorm2d层模式python实现代码示例:
            模式一：
            bn_conv1 = self.batchnorm0(conv1)
            scale_conv1_cparam1 = self.scale_conv1_cparam1
            scale_conv1_mul = paddle.multiply(x=bn_conv1, y=scale_conv1_cparam1, axis=1)
            scale_conv1_cparam2 = self.scale_conv1_cparam2
            scale_conv1 = paddle.add(x=scale_conv1_mul, y=scale_conv1_cparam2, axis=1)
            模式二：
            bn_conv1 = self.batchnorm0(conv1)
            scale_conv1_cparam1 = self.scale_conv1_cparam1
            scale_conv1_mul = paddle.multiply(x=bn_conv1, y=scale_conv1_cparam1, axis=1)
            scale_conv1_cparam2 = self.scale_conv1_cparam2
            scale_conv1_cparam2 = paddle.reshape(x=scale_conv1_cparam2, shape=[32, 1, 1])
            scale_conv1 = paddle.add(x=scale_conv1_mul, y=scale_conv1_cparam2, axis=1)
        """

        def gen_name(id):
            return "x" + str(id)

        pattern = PaddleGraph()
        pattern.add_layer(
            "paddle.nn.BatchNorm2D",
            inputs={"input": "bn-input-0"},
            outputs=[gen_name(0)])
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(1)])
        inputs_dict = {}
        inputs_dict['x'] = gen_name(0)
        inputs_dict['y'] = gen_name(1)
        pattern.add_layer(
            "paddle.multiply", inputs=inputs_dict, outputs=[gen_name(2)])
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(3)])
        inputs_dict = {}
        inputs_dict['x'] = gen_name(2)
        inputs_dict['y'] = gen_name(3)
        pattern.add_layer(
            "paddle.add", inputs=inputs_dict, outputs=[gen_name(4)])
        pattern.build(inputs={"input-0": "bn-input-0"})
        self.patterns.append(pattern)

        pattern = PaddleGraph()
        pattern.add_layer(
            "paddle.nn.BatchNorm2D",
            inputs={"input": "bn-input-0"},
            outputs=[gen_name(0)])
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(1)])
        inputs_dict = {}
        inputs_dict['x'] = gen_name(0)
        inputs_dict['y'] = gen_name(1)
        pattern.add_layer(
            "paddle.multiply", inputs=inputs_dict, outputs=[gen_name(2)])
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(3)])
        pattern.add_layer(
            "paddle.reshape", inputs={"x": gen_name(3)}, outputs=[gen_name(3)])
        inputs_dict = {}
        inputs_dict['x'] = gen_name(2)
        inputs_dict['y'] = gen_name(3)
        pattern.add_layer(
            "paddle.add", inputs=inputs_dict, outputs=[gen_name(4)])
        pattern.build(inputs={"input-0": "bn-input-0"})
        self.patterns.append(pattern)

    def insert_new_layer(self, graph, parameters, matches):
        new_layer = self.gen_new_layer(parameters, matches)
        new_layer_id = list(matches.keys())[0]
        graph.layers[new_layer_id] = new_layer
        matches.pop(new_layer_id)

    def gen_new_layer(self, parameters, matches):
        layers_id = list(matches.keys())
        layer = matches[layers_id[0]]
        layer_inputs = layer.inputs
        bn_name = layer.outputs[0]
        layer_attrs = layer.attrs
        layer_attrs.pop("weight_attr")
        layer_attrs.pop("bias_attr")
        layer = matches[layers_id[-1]]
        layer_outputs = [bn_name] + layer.outputs
        layer = matches[layers_id[1]]
        data0_name = layer.outputs[0]
        data0_numpy = parameters.pop(data0_name)
        parameters["{}.weight".format(layer_outputs[0])] = data0_numpy
        layer = matches[layers_id[3]]
        data1_name = layer.outputs[0]
        data1_numpy = parameters.pop(data1_name)
        parameters["{}.bias".format(layer_outputs[0])] = data1_numpy
        new_layer = PaddleLayer(
            layers_id[0],
            "paddle.nn.BatchNorm2D",
            inputs=layer_inputs,
            outputs=layer_outputs,
            **layer_attrs)
        return new_layer
