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
from x2paddle.optimizer.pytorch_optimizer.pattern_matcher import FuseBase
from x2paddle.core.program import PaddleGraph, PaddleLayer
from x2paddle.core.util import *


class TraceFcFuser(FuseBase):
    def __init__(self):
        self.linear_index = 0
        super(TraceFcFuser, self).__init__(graph_type="dygraph")
        
    def build_pattern(self):
        """ 描述需要替换的fc图结构。
        fc层模式python实现代码示例:
           encoder_layer_8_attention_self_key_weight = self.encoder_layer_8_attention_self_key_weight
           x748 = fluid.layers.transpose(x=encoder_layer_8_attention_self_key_weight, perm=[1, 0])
           x749 = paddle.matmul(x=x732, y=x748)
           encoder_layer_8_attention_self_key_bias = self.encoder_layer_8_attention_self_key_bias
           x750 = x749 + 1 * encoder_layer_8_attention_self_key_bias
        """

        def gen_name(id):
            return "x" + str(id)

        self.pattern.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[gen_name(0)],
            value="params[{}]".format(string(gen_name(0))))
        self.pattern.add_layer(
            "fluid.layers.transpose",
            inputs={"x": gen_name(0)},
            outputs=[gen_name(1)],
            perm=[1, 0])
        self.pattern.add_layer(
            "paddle.matmul",
            inputs={"x": "fc-input-0",
                    "y": gen_name(1)},
            outputs=[gen_name(2)])
        self.pattern.add_layer(
            "fluid.dygraph.base.to_variable",
            inputs={},
            outputs=[gen_name(3)],
            value="params[{}]".format(string(gen_name(3))))
        self.pattern.add_layer(
            "prim.add_",
            inputs={"x": gen_name(2),
                    "y": gen_name(3)},
            outputs=[gen_name(4)],
            alpha=1)
        self.pattern.build(inputs={"input-0": "fc-input-0"})

    def insert_new_layer(self, graph, parameters, matches):
        new_layer = self.gen_new_layer(parameters, matches)
        new_layer_id = list(matches.keys())[0]
        graph.layers[new_layer_id] = new_layer
        matches.pop(new_layer_id)

    def gen_new_layer(self, parameters, matches):
        layers_id = list(matches.keys())
        layer = matches[layers_id[2]]
        input_name = layer.inputs["x"]
        scope_name = layer.scope_name
        layer = matches[layers_id[4]]
        output_name = layer.outputs[0]
        layer = matches[layers_id[0]]
        weight_name = layer.attrs["value"][8:-2]
        layer = matches[layers_id[3]]
        bias_name = layer.attrs["value"][8:-2]
        attrs = dict()
        attrs["in_features"] = parameters[weight_name].shape[1]
        attrs["out_features"] = parameters[weight_name].shape[0]
        linear_name = "linear{}".format(self.linear_index)
        self.linear_index += 1
        parameters["{}.weight".format(linear_name)] = parameters[
            weight_name].transpose((1, 0))
        parameters["{}.bias".format(linear_name)] = np.squeeze(parameters[
            bias_name])
        new_layer = PaddleLayer(
            layers_id[0],
            "paddle.nn.Linear",
            inputs={"input": input_name},
            outputs=[linear_name, output_name],
            scope_name=scope_name,
            **attrs)
        return new_layer

