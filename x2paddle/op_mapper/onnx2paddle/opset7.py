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

from .opset_legacy import OpSet
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


class OpSet7(OpSet):
    def __init__(self, decoder, paddle_graph):
        super(OpSet7, self).__init__(decoder, paddle_graph)

    @print_mapping_info
    def Or(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        self.paddle_graph.add_layer(
            "paddle.logical_or",
            inputs={"x": val_x.name,
                    "y": val_y.name},
            outputs=[node.name])

    @print_mapping_info
    def Xor(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        self.paddle_graph.add_layer(
            "paddle.logical_xor",
            inputs={"x": val_x.name,
                    "y": val_y.name},
            outputs=[node.name])

    @print_mapping_info
    def Unsqueeze(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = node.get_attr('axes')
        # deal with scalar(0D) tensor
        if len(val_x.out_shapes[0]) == 0 and len(axes) == 1 and axes[0] == 0:
            self.paddle_graph.add_layer(
                'paddle.reshape',
                inputs={"x": val_x.name},
                outputs=[node.name],
                shape=[1])
        else:
            self.paddle_graph.add_layer(
                'paddle.unsqueeze',
                inputs={"x": val_x.name},
                axis=axes,
                outputs=[node.name])

    @print_mapping_info
    def ReduceL1(self, node):
        output_name = node.name
        layer_outputs = [output_name]
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = node.get_attr('axes')
        keepdims = False if node.get_attr('keepdims') == 0 else True
        layer_attrs = {'p': 1, 'axis': axes, 'keepdim': keepdims}
        if val_x.dtype == 'int32':

            indices_cast = val_x.name + '_cast'
            mid_norm = val_x.name + '_norm'

            self.paddle_graph.add_layer(
                'paddle.cast',
                inputs={"x": val_x.name},
                outputs=[indices_cast],
                dtype=string('float32'))
            self.paddle_graph.add_layer(
                "paddle.norm",
                inputs={"x": indices_cast},
                outputs=[mid_norm],
                **layer_attrs)
            self.paddle_graph.add_layer(
                'paddle.cast',
                inputs={"x": mid_norm},
                outputs=[node.name],
                dtype=string(val_x.dtype))
        else:
            self.paddle_graph.add_layer(
                "paddle.norm",
                inputs={"x": val_x.name},
                outputs=layer_outputs,
                **layer_attrs)

    @print_mapping_info
    def ReduceL2(self, node):
        output_name = node.name
        layer_outputs = [output_name]
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = node.get_attr('axes')
        keepdims = False if node.get_attr('keepdims') == 0 else True
        layer_attrs = {'p': 2, 'axis': axes, 'keepdim': keepdims}
        if val_x.dtype == 'int32':
            indices_cast = val_x.name + '_cast'
            mid_norm = val_x.name + '_norm'

            self.paddle_graph.add_layer(
                'paddle.cast',
                inputs={"x": val_x.name},
                outputs=[indices_cast],
                dtype=string('float32'))
            self.paddle_graph.add_layer(
                "paddle.norm",
                inputs={"x": indices_cast},
                outputs=[mid_norm],
                **layer_attrs)
            self.paddle_graph.add_layer(
                'paddle.cast',
                inputs={"x": mid_norm},
                outputs=[node.name],
                dtype=string(val_x.dtype))
        else:
            self.paddle_graph.add_layer(
                "paddle.norm",
                inputs={"x": val_x.name},
                outputs=layer_outputs,
                **layer_attrs)
