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

from .opset12 import OpSet12


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


class OpSet13(OpSet12):

    def __init__(self, decoder, paddle_graph):
        super(OpSet13, self).__init__(decoder, paddle_graph)

    @print_mapping_info
    def Unsqueeze(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = self.graph.get_input_node(node, idx=1, copy=True)
        # deal with scalar(0D) tensor
        if len(val_x.out_shapes[0]) == 0 and len(
                axes.out_shapes[0]) == 1 and len(node.out_shapes[0]) == 1:
            self.paddle_graph.add_layer('paddle.reshape',
                                        inputs={"x": val_x.name},
                                        outputs=[node.name],
                                        shape=[1])
        else:
            self.paddle_graph.add_layer('paddle.unsqueeze',
                                        inputs={
                                            "x": val_x.name,
                                            "axis": axes.name
                                        },
                                        outputs=[node.name])
