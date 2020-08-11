#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

from x2paddle.optimizer.linear_pass import LinearPass, LinearMatcher


class GraphOptimizer(object):
    def __init__(self):
        linear_pass = LinearPass()
        linear_matcher = LinearMatcher()
        self.passes = {linear_pass: linear_matcher}

    def run(self, graph):
        is_update_graph = False
        while True:
            for i, (layer_id, layer) in enumerate(graph.layers.items()):
                is_match = self.current_matcher.match_pattern(
                    self.current_pass.pattern, graph, i)
                if is_match:
                    is_update_graph = True
                    graph = self.current_matcher.replace_layer(graph, is_match)
                    break
                for j, block in enumerate(layer.blocks):
                    if len(block.layers) > 0:
                        layer.blocks[j], is_update_block = self.run(block)
                        if is_update_block:
                            break
                if i + 1 == len(graph.layers):
                    return graph, is_update_graph

    def optimize(self, graph):
        # 开始优化
        for pa, ma in self.passes.items():
            self.current_pass = pa
            self.current_matcher = ma
            graph, _ = self.run(graph)
            print("{} done!".format(pa.__class__.__name__))
        return graph
