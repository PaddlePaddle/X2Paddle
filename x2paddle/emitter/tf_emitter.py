#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from x2paddle.parser.tf_parser import TFGraph
from x2paddle.core.emitter import Emitter
from x2paddle.core.fluid_code import FluidCode
from x2paddle.core.util import *


class TFEmitter(Emitter):
    def __init__(self, parser):
        super(TFEmitter, self).__init__()
        self.parser = parser
        self.graph = parser.tf_graph
        self.weights = dict()

    def run(self):
        print("Total nodes: {}".format(len(self.graph.topo_sort)))
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if hasattr(self, op):
                emit_func = getattr(self, op)
                emit_func(node)

        for i in range(len(self.graph.topo_sort)):
            node_name = self.graph.topo_sort[i]
            node = self.graph.get_node(node_name)
            for layer in node.fluid_code.layers:
                print(layer.get_code())

    def Placeholder(self, node):
        shape = node.out_shapes[0]
        dtype = node.dtype
        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer("data",
                                  inputs=None,
                                  output=node,
                                  param_attr=attr)

    def Const(self, node):
        shape = node.out_shapes[0]
        dtype = node.dtype
        value = node.value
        initializer = "Constant(0.0)"
        if len(shape) == 0:
            assert value.size == 1, "Unexpected situation happend"
            shape = [1]
            initializer = "Constant({})".format(value)

        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name),
            'default_initializer': initializer
        }
        node.fluid_code.add_layer("create_parameter",
                                  inputs=None,
                                  output=node,
                                  param_attr=attr)

    def Transpose(self, node):
        input = self.graph.get_node(node.layer.input[0])
        perm = self.graph.get_node(node.layer.input[1])
        perm.fluid_code.clear()
        perm = perm.value.tolist()

        attr = {'perm': perm}
        node.fluid_code.add_layer("transpose",
                                  inputs=input,
                                  output=node,
                                  param_attr=attr)

    def RealDiv(self, node):
        x = self.graph.get_node(node.layer.input[0])
        y = self.graph.get_node(node.layer.input[1])
        inputs = {'x': x, 'y': y}
        node.fluid_code.add_layer("elementwise_div",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=None)

    def Fc(self, node):
        self.weight['asdf'] = np.tranpose(node.kerneln[1, 0])
