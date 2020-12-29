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


class Static_BNScaleFuser(FuseBase):
    def __init__(self):
        super(Static_BNScaleFuser, self).__init__(graph_type="static")
        self.patterns = list()

    def build_pattern(self):
        """ 描述需要替换的batchnorm2d图结构。
        batchnorm2d层模式python实现代码示例:
        模式一：
        conv1_bn_mean = paddle.static.create_parameter(shape=(128,), dtype='float32', name='conv1_bn_mean')
        conv1_bn_variance = paddle.static.create_parameter(shape=(128,), dtype='float32', name='conv1_bn_variance')
        conv1_bn = paddle.nn.functional.batch_norm(x=conv1, weight=conv1_bn_weight, bias=conv1_bn_bias, running_mean=conv1_bn_mean, running_var=conv1_bn_variance, epsilon=9.999999747378752e-06, momentum=0.9990000128746033)
        conv1_scale_cparam1 = paddle.static.create_parameter(shape=(32,), dtype='float32', name='conv1_scale_cparam1')
        conv1_scale_mul = paddle.multiply(x=conv1_bn, y=conv1_scale_cparam1, axis=1)
        conv1_scale_cparam2 = paddle.static.create_parameter(shape=(32,), dtype='float32', name='conv1_scale_cparam2')
        conv1_scale_cparam2 = paddle.reshape(x=conv1_scale_cparam2, shape=[32, 1, 1])
        conv1_scale = paddle.add(x=conv1_scale_mul, y=conv1_scale_cparam2)
        模式二：
        conv1_bn_mean = paddle.static.create_parameter(shape=(128,), dtype='float32', name='conv1_bn_mean')
        conv1_bn_variance = paddle.static.create_parameter(shape=(128,), dtype='float32', name='conv1_bn_variance')
        conv1_bn = paddle.nn.functional.batch_norm(x=conv1, weight=conv1_bn_weight, bias=conv1_bn_bias, running_mean=conv1_bn_mean, running_var=conv1_bn_variance, epsilon=9.999999747378752e-06, momentum=0.9990000128746033)
        conv1_scale_cparam1 = paddle.static.create_parameter(shape=(32,), dtype='float32', name='conv1_scale_cparam1')
        conv1_scale_mul = paddle.multiply(x=conv1_bn, y=conv1_scale_cparam1, axis=1)
        conv1_scale_cparam2 = paddle.static.create_parameter(shape=(32,), dtype='float32', name='conv1_scale_cparam2')
        conv1_scale = paddle.add(x=conv1_scale_mul, y=conv1_scale_cparam2)
        """

        def gen_name(id):
            return "x" + str(id)
        
        pattern = PaddleGraph(graph_type="dygraph")
        pattern.add_layer(
            "paddle.static.create_parameter",
            inputs={},
            outputs=[gen_name(10)])
        pattern.add_layer(
            "paddle.static.create_parameter",
            inputs={},
            outputs=[gen_name(11)])
        pattern.add_layer(
            "paddle.nn.functional.batch_norm",
            inputs={"input": "bn-input-0",
                    "weight": "bn-input-1",
                    "bias": "bn-input-2",
                    "running_mean": gen_name(10),
                    "running_var": gen_name(11)},
            outputs=[gen_name(0)])
        pattern.add_layer(
            "paddle.static.create_parameter",
            inputs={},
            outputs=[gen_name(1)])
        inputs_dict = {}
        inputs_dict['x'] = gen_name(0)
        inputs_dict['y'] = gen_name(1)
        pattern.add_layer(
            "paddle.multiply",
            inputs=inputs_dict,
            outputs=[gen_name(2)])
        pattern.add_layer(
            "paddle.static.create_parameter",
            inputs={},
            outputs=[gen_name(3)])
        pattern.add_layer(
            "paddle.reshape",
            inputs={"x": gen_name(3)},
            outputs=[gen_name(4)])
        inputs_dict = {}
        inputs_dict['x'] = gen_name(2)
        inputs_dict['y'] = gen_name(4)
        pattern.add_layer(
            "paddle.add",
            inputs=inputs_dict,
            outputs=[gen_name(5)])
        pattern.build(inputs={"input-0": "bn-input-0",
                              "input-1": "bn-input-1",
                              "input-2": "bn-input-2"})
        self.patterns.append(pattern)
        
        pattern = PaddleGraph(graph_type="dygraph")
        pattern.add_layer(
            "paddle.static.create_parameter",
            inputs={},
            outputs=[gen_name(10)])
        pattern.add_layer(
            "paddle.static.create_parameter",
            inputs={},
            outputs=[gen_name(11)])
        pattern.add_layer(
            "paddle.nn.functional.batch_norm",
            inputs={"input": "bn-input-0",
                    "weight": "bn-input-1",
                    "bias": "bn-input-2",
                    "running_mean": gen_name(10),
                    "running_var": gen_name(11),},
            outputs=[gen_name(0)])
        pattern.add_layer(
            "paddle.static.create_parameter",
            inputs={},
            outputs=[gen_name(1)])
        inputs_dict = {}
        inputs_dict['x'] = gen_name(0)
        inputs_dict['y'] = gen_name(1)
        pattern.add_layer(
            "paddle.multiply",
            inputs=inputs_dict,
            outputs=[gen_name(2)])
        pattern.add_layer(
            "paddle.static.create_parameter",
            inputs={},
            outputs=[gen_name(3)])
        inputs_dict = {}
        inputs_dict['x'] = gen_name(2)
        inputs_dict['y'] = gen_name(3)
        pattern.add_layer(
            "paddle.add",
            inputs=inputs_dict,
            outputs=[gen_name(4)])
        pattern.build(inputs={"input-0": "bn-input-0",
                              "input-1": "bn-input-1",
                              "input-2": "bn-input-2"})
        self.patterns.append(pattern)

    def insert_new_layer(self, graph, parameters, matches):
        new_layer = self.gen_new_layer(parameters, matches)
        new_layer_id = list(matches.keys())[-1]
        graph.layers[new_layer_id] = new_layer
        matches.pop(list(matches.keys())[0])
        matches.pop(list(matches.keys())[0])
        matches.pop(list(matches.keys())[1])
        matches.pop(list(matches.keys())[2])
        matches.pop(new_layer_id)

    def gen_new_layer(self, parameters, matches):
        layers_id = list(matches.keys())
        bn_layer = matches[layers_id[2]]
        layer = matches[layers_id[3]]
        bn_layer.inputs["weight"] = layer.outputs[0]
        layer = matches[layers_id[5]]
        bn_layer.inputs["bias"] = layer.outputs[0]
        bn_layer.id = layers_id[-1]
        layer = matches[layers_id[-1]]
        bn_layer.outputs = layer.outputs
        return bn_layer