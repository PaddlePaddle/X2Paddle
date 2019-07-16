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
        self.identity_ops = ['Identity']

    def remove_isolated_node(self, graph):
        # delete isolated nodes
        isolated_nodes = list()
        for node_name in graph.node_map.keys():
            if len(graph.get_node(node_name).inputs) == 0 or len(
                    graph.get_node(node_name).outputs) == 0:
                isolated_nodes.append(node_name)

        graph.remove_node(node_name)

    def run(self, graph):
        self.remove_isolated_node(graph)


# TODO identity node remove

# TODO subgraph optimize

# TODO compute optimize
