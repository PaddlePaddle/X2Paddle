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

from x2paddle.core.graph import GraphNode, Graph


class TFGraphNode(GraphNode):
    def __init__(self, layer, layer_name=None):
        super(TFGraphNode, self).__init__(layer, layer_name)
        self.layer_type = layer.op.lower()


class TFGraph(Graph):
    def __init__(self, model):
        super(TFGraph, self).__init__(model)


class TFParser(object):
    def __init__(self, pb_model, in_nodes=None, out_nodes=None, in_shapes=None):
        assert in_nodes is not None, "in_nodes should not be None"
        assert out_nodes is not None, "out_nodes should not be None"
        assert in_shapes is not None, "in_shapes should not be None"
        assert len(in_shapes) == len(in_nodes), "length of in_shapes and in_nodes should be equal"

        serialized_str = open(pb_model, 'rb').read()
        tf.reset_default_graph()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(serialized_str)

        sess = tf.Session(graph=tf.get_default_graph())
        sess.run(tf.global_variables_initializer())
        
