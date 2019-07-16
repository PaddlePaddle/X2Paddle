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


class TFEmitter(Emitter):
    def __init__(self, parser):
        super(TFEmitter, self).__init__()
        self.parser = parser
        self.graph = parser.tf_graph
        self.fluid_code = FluidCode()

    def run(self):
        print("Total nodes: {}".format(len(self.graph.topo_sort)))
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if hasattr(self, op):
                emit_func = getattr(self, op)
                emit_func(node)

    def Placeholder(self, node):
        shape = node.out_shapes[0]
        dtype = node.dtype
        attr = {
            'dtype': '\{}\''.format(dtype),
            'shape': shape,
            'name': '\'{}\''.format(node.layer_name)
        }
        self.fluid_code.add_layer("data",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)
        print(self.fluid_code.layers[0].get_code())
