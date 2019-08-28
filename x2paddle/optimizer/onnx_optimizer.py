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

# TODO useless node remove
from x2paddle.op_mapper.onnx_op_mapper import ONNXOpMapper


class ONNXOptimizer(object):
    def __init__(self, op_mapper):
        self.op_mapper = op_mapper
        self.graph = op_mapper.graph

    def delete_redundance_code(self):
        for node_name in self.graph.topo_sort:
            if node_name in self.op_mapper.omit_nodes:
                node = self.graph.get_node(node_name)
                omit_freq = self.op_mapper.omit_nodes.count(node_name)
                if len(node.outputs) <= omit_freq:
                    node.fluid_code.clear()
