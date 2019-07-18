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
        print("Doint Nothing")

    def remove_isolated_node(self, graph):
        # delete isolated nodes
        isolated_nodes = list()
        for node_name in graph.node_map.keys():
            if len(graph.get_node(node_name).inputs) == 0 or len(
                    graph.get_node(node_name).outputs) == 0:
                isolated_nodes.append(node_name)

        graph.remove_node(node_name)

    def remove_identity_node(self, graph):
        identity_node = list()
        for node_name, node in graph.node_map.items():
            if node.layer_type == "Identity":
                identity_node.append(node_name)

        for node_name in identity_node:
            node = graph.get_node(node_name)
            # Remind: Only 1 input for Identity node
            input_node = graph.get_node(node.inputs[0])

            # remove identity node from graph
            idx = input_node.outputs.index(node_name)
            del input_node.outputs[idx]

            output_names = node.outputs
            for output_name in output_names:
                output_node = graph.get_node(output_name)
                idx = output_node.inputs.index(node_name)
                output_node.inputs[idx] = input_node.layer_name

            idx = graph.topo_sort.index(node_name)
            del graph.topo_sort[idx]

    def run(self, graph):
        self.remove_isolated_node(graph)
        self.remove_identity_node(graph)


# TODO identity node remove

# TODO subgraph optimize

# TODO compute optimize

# activation merge

# biasadd merge
