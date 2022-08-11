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

from .opset9 import OpSet9


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


class OpSet10(OpSet9):
    def __init__(self, decoder, paddle_graph):
        super(OpSet10, self).__init__(decoder, paddle_graph)
        # Support Mod op Since opset version >= 10
        self.elementwise_ops.update({"Mod": "paddle.mod"})

    @print_mapping_info
    def IsInf(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        if node.get_attr('detect_negative') != None or node.get_attr(
                'detect_positive') != None:
            if node.get_attr('detect_negative') != 1 or node.get_attr(
                    'detect_positive') != 1:
                raise Exception(
                    "x2addle does not currently support IsINF with attributes 'detect_negative' and 'detect_positive'."
                )
        else:
            self.paddle_graph.add_layer(
                'paddle.isinf', inputs={"x": val_x.name}, outputs=[node.name])
