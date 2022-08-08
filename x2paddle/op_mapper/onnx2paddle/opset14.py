# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

from .opset13 import OpSet13
from x2paddle.core.util import *


def print_mapping_info(func):
    def run_mapping(*args, **kwargs):
        node = args[1]
        try:
            res = func(*args, **kwargs)
        except:
            raise Exception("convert failed node:{}, op_type is {}".format(
                node.name[9:], node.layer_type))
        else:
            return res

    return run_mapping


class OpSet14(OpSet13):
    def __init__(self, decoder, paddle_graph):
        super(OpSet14, self).__init__(decoder, paddle_graph)

    @print_mapping_info
    def Relu(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)

        # if val_x.dtypes!='float':
        indices_cast = val_x.name + '_cast'
        mid_relu = val_x.name + '_relu'
        self.paddle_graph.add_layer(
            'paddle.cast',
            inputs={"x": val_x.name},
            outputs=[indices_cast],
            dtype=string('float32'))
        self.paddle_graph.add_layer(
            'paddle.nn.ReLU', inputs={"x": indices_cast}, outputs=[mid_relu])
        self.paddle_graph.add_layer(
            'paddle.cast',
            inputs={"x": mid_relu},
            outputs=[node.name],
            dtype=string(val_x.dtype))
