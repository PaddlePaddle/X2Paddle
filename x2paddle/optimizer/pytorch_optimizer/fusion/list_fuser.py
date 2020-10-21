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


class ListFuser(FuseBase):
    def __init__(self):
        super(ListFuser, self).__init__(graph_type="dygraph")

    def build_pattern(self):
        """ 描述需要替换的constant图结构。
        list层模式python实现代码示例:
            p_size = [3, 3]
            pool1 = paddle.nn.Pool2D(pool_size=p_size, pool_stride=[2, 2], pool_padding=[0, 0], ceil_mode=False, pool_type='max')
        """
        self.pattern.add_layer(
            "prim.list", inputs={}, outputs=["x1"], input0=3, input1=3)
        self.pattern.build()
        self.pattern.outputs = ["x1"]

    def insert_new_layer(self, graph, parameters, matches):
        def replace_value(layer_connect, match_name, match_value):
            for k, v in layer_connect.inputs.items():
                if v == match_name:
                    layer_connect.inputs.pop(k)
                    layer_connect.attrs[k] = match_value
                    break
            for k, v in layer_connect.attrs.items():
                if v == match_name:
                    layer_connect.attrs[k] = match_value
                    break
            if layer_connect.kernel == "prim.loop" or \
            layer_connect.kernel == "prim.if":
                for block in layer_connect.blocks:
                    for b_layer_id, b_layer in block.layers.items():
                        if block.edges_in.get(b_layer_id, 0) != 0 and  \
                        -1 in block.edges_in[b_layer_id]:
                            replace_value(b_layer, match_name, match_value)
                            
        def get_value(layer, key):
            """ 获取list的组成。
            """
            if key in layer.inputs:
                return layer.inputs[key]
            else:
                return str(layer.attrs[key])

        layer_id = list(matches.keys())[0]
        layer = list(matches.values())[0]
        layer_output_name = layer.outputs[0]
        
        input_len = len(layer.inputs) + len(layer.attrs)
        inputs_list = list()
        for i in range(input_len):
            inputs_list.append(get_value(layer, "input{}".format(i)))
        inputs_str = ", ".join(inputs_list)
        inputs_str = "[" + inputs_str + "]"
        
        layer_value = inputs_str
        if graph.edges_out.get(layer_id, 0) != 0:
            for layer_id_out in graph.edges_out[layer_id]:
                layer_connect = graph.layers[layer_id_out]
                replace_value(layer_connect, layer_output_name, layer_value)