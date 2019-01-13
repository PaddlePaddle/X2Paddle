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

from name_generator import NameGenerator


class GraphNode(object):
    def __init__(self, layer):
        self.in_edges = list()
        self.out_edges = list()
        self.layer = layer
        self.ref_name = None


class Graph(object):
    def __init__(self, model):
        self.node_map = dict()
        self.input_nodes = list()
        self.output_nodes = list()
        self.topological_sort = list()
        self.model = model
        self.name_generator = NameGenerator()

    def build(self):
        self._make_input_nodes()
        self._make_output_nodes()
        self._get_topological_sort()
        self._gen_newname_for_nodes()

    def _make_input_nodes(self):
        for name, node in self.node_map.items():
            node.left_in_edges = len(node.in_edges)
            if len(node.in_edges) == 0:
                self.input_nodes.append(name)

    def _make_output_nodes(self):
        for name, node in self.node_map.items():
            if len(node.out_edges) == 0:
                self.output_nodes.append(name)

    def _get_topological_sort(self):
        self.topological_sort = self.input_nodes[:]
        idx = 0
        while idx < len(self.topological_sort):
            current_node = self.node_map[self.topological_sort[idx]]
            for next_node in current_node.out_edges:
                next_node_info = self.node_map[next_node]
                next_node_info.left_in_edges -= 1
                if next_node_info.left_in_edges == 0:
                    self.topological_sort.append(next_node)
            idx += 1

    def _gen_newname_for_nodes(self):
        for node_name in self.topological_sort:
            node = self.node_map[node_name]
            ref_name = self.name_generator.get_name(node)
            self.node_map[node.layer.name].ref_name = ref_name

    def get_node(self, name):
        if name not in self.node_map:
            raise Exception("Graph doesn't have node [%s]." % name)
        else:
            return self.node_map[name]

    def _make_connection(self, src, dst):
        if src == dst or src not in self.node_map or dst not in self.node_map:
            raise Exception('Warning: Node not exist or there is a self-loop')
        if src not in self.node_map[dst].in_edges:
            self.node_map[dst].in_edges.append(src)
        if dst not in self.node_map[src].out_edges:
            self.node_map[src].out_edges.append(dst)
