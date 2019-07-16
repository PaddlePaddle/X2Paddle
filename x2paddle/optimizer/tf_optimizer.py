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
from x2paddle.parser.tf_parser import TFGraph


class TFGraphOptimizer(object):
    def __init__(self):
        print("Not Implement")
        self.useless_op = [
                'NoOp']

    def remove_useless_node(self, graph):
        for name, node in graph.node_map.items():
            if node.layer_type in self.useless_op:

# TODO identity node remove

# TODO subgraph optimize

# TODO compute optimize
