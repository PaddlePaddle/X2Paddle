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


class TFBatchNormFuser(FuseBase):
    def __init__(self):
        self.bn_index = 0
        super(TFBatchNormFuser, self).__init__()
        self.patterns = list()

    def build_pattern(self):
        """ 描述需要替换的batchnorm图结构。
        batchnorm层模式python实现代码示例:

        """

        def gen_name(id):
            return "x" + str(id)

        pattern = PaddleGraph()
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(0)])
        pattern.add_layer(
            "paddle.full", inputs={}, outputs=[gen_name(1)], shape=[1])
        pattern.add_layer(
            "paddle.add",
            inputs={"x": gen_name(0),
                    "y": gen_name(1)},
            outputs=[gen_name(2)])
        pattern.add_layer(
            "paddle.rsqrt", inputs={"x": gen_name(2)}, outputs=[gen_name(3)])
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(4)])
        pattern.add_layer(
            "paddle.multiply",
            inputs={"x": gen_name(3),
                    "y": gen_name(4)},
            outputs=[gen_name(5)])
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(6)])
        pattern.add_layer(
            "paddle.multiply",
            inputs={"x": gen_name(6),
                    "y": gen_name(5)},
            outputs=[gen_name(7)])
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(8)])
        pattern.add_layer(
            "paddle.subtract",
            inputs={"x": gen_name(8),
                    "y": gen_name(7)},
            outputs=[gen_name(9)])
        pattern.add_layer(
            "paddle.multiply",
            inputs={"x": "bn-input-0",
                    "y": gen_name(5)},
            outputs=[gen_name(10)])
        pattern.add_layer(
            "paddle.add",
            inputs={"x": gen_name(10),
                    "y": gen_name(9)},
            outputs=[gen_name(11)])
        pattern.build(inputs={"input-0": "bn-input-0", })
        self.patterns.append(pattern)

        pattern = PaddleGraph()
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(0)])
        pattern.add_layer(
            "paddle.full", inputs={}, outputs=[gen_name(1)], shape=[1])
        pattern.add_layer(
            "paddle.add",
            inputs={"x": gen_name(0),
                    "y": gen_name(1)},
            outputs=[gen_name(2)])
        pattern.add_layer(
            "paddle.rsqrt", inputs={"x": gen_name(2)}, outputs=[gen_name(3)])
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(4)])
        pattern.add_layer(
            "paddle.multiply",
            inputs={"x": gen_name(3),
                    "y": gen_name(4)},
            outputs=[gen_name(5)])
        pattern.add_layer(
            "paddle.multiply",
            inputs={"x": "bn-input-0",
                    "y": gen_name(5)},
            outputs=[gen_name(10)])
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(6)])
        pattern.add_layer(
            "paddle.multiply",
            inputs={"x": gen_name(6),
                    "y": gen_name(5)},
            outputs=[gen_name(7)])
        pattern.add_layer(
            "self.create_parameter", inputs={}, outputs=[gen_name(8)])
        pattern.add_layer(
            "paddle.subtract",
            inputs={"x": gen_name(8),
                    "y": gen_name(7)},
            outputs=[gen_name(9)])
        pattern.add_layer(
            "paddle.add",
            inputs={"x": gen_name(10),
                    "y": gen_name(9)},
            outputs=[gen_name(11)])
        pattern.build(inputs={"input-0": "bn-input-0", })
        self.patterns.append(pattern)

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
            if layer.kernel == "paddle.full":
                full_layer = layer
                out_layer_id = graph.edges_out[layer_id][0]
                if matches[out_layer_id].kernel == "paddle.add":
                    var_layer_id = graph.edges_in[out_layer_id][0]
                    var_layer = matches[var_layer_id]
            if layer.kernel == "paddle.rsqrt":
                out_layer_id = graph.edges_out[layer_id][0]
                if matches[out_layer_id].kernel == "paddle.multiply":
                    gamma_layer_id = graph.edges_in[out_layer_id][1]
                    gamma_layer = matches[gamma_layer_id]
            if layer.kernel == "paddle.subtract":
                in_layer_id = graph.edges_in[layer_id][0]
                beta_layer = matches[in_layer_id]
                in_layer_id = graph.edges_in[layer_id][1]
                in_layer_id = graph.edges_in[in_layer_id][0]
                mean_layer = matches[in_layer_id]
                out_layer_id = graph.edges_out[layer_id][0]
                add_layer = matches[out_layer_id]
            if layer.kernel == "paddle.multiply":
                in_layer_id = graph.edges_in[layer_id][1]
                mul_layer = matches[in_layer_id]
                if mul_layer.kernel == "paddle.multiply":
                    in_layer_id = graph.edges_in[layer_id][0]
                    if in_layer_id not in matches:
                        input_name = layer.inputs["x"]
        transpose0 = PaddleLayer(
            id=layer_id_list[-1] + "_1",
            kernel="paddle.transpose",
            inputs={"x": input_name},
            outputs=["{}_transpose_for_bn".format(input_name)],
            perm=[0, 3, 1, 2])
        bn_name = "merge_bn{}".format(self.bn_index)
        self.bn_index += 1
        params = parameters[gamma_layer.outputs[0]]
        c = params.shape[0]
        bn = PaddleLayer(
            id=layer_id_list[-1] + "_2",
            kernel="paddle.nn.BatchNorm",
            inputs={"input": "{}_transpose_for_bn".format(input_name)},
            outputs=[bn_name, "{}_bn".format(input_name)],
            num_channels=c,
            epsilon=full_layer.attrs["fill_value"],
            param_attr=string(gamma_layer.outputs[0]),
            bias_attr=string(beta_layer.outputs[0]),
            moving_mean_name=string(mean_layer.outputs[0]),
            moving_variance_name=string(var_layer.outputs[0]),
            is_test=True)
        transpose1 = PaddleLayer(
            id=layer_id_list[-1] + "_3",
            kernel="paddle.transpose",
            inputs={"x": "{}_bn".format(input_name)},
            outputs=add_layer.outputs,
            perm=[0, 2, 3, 1])
        return [transpose0, bn, transpose1], layer_id_list[-1]
