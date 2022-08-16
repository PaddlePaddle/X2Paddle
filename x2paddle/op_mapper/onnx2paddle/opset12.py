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

from .opset11 import OpSet11


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


class OpSet12(OpSet11):
    def __init__(self, decoder, paddle_graph):
        super(OpSet12, self).__init__(decoder, paddle_graph)

    @print_mapping_info
    def ArgMin(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axis = node.get_attr('axis')
        keepdims = False if node.get_attr('keepdims') == 0 else True
        select_last_index = node.get_attr('select_last_index')
        if select_last_index != 0:
            raise Exception(
                "Argmin operator conversion when select_last_index is equal to 1 is currently not supported."
            )
        layer_attrs = {
            'axis': axis,
            'keepdim': keepdims,
        }
        if select_last_index == 0:
            self.paddle_graph.add_layer(
                'paddle.argmin',
                inputs={"x": val_x.name},
                outputs=[node.name],
                **layer_attrs)
