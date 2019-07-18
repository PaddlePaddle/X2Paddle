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
from x2paddle.core.fluid_code import FluidCode
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.core.framework import attr_value_pb2
import tensorflow as tf
import copy


class TFGraphNode(GraphNode):
    def __init__(self, layer, layer_name=None):
        if layer_name is None:
            super(TFGraphNode, self).__init__(layer, layer.name)
        else:
            super(TFGraphNode, self).__init__(layer, layer_name)

        self.layer_type = layer.op
        self.fluid_code = FluidCode()

        self.dtype_map = {1: "float32", 3: "int32", 9: "int64"}

    @property
    def out_shapes(self):
        values = self.layer.attr["_output_shapes"].list.shape
        out_shapes = list()
        for value in values:
            shape = [dim.size for dim in value.dim]
            out_shapes.append(shape)
        return out_shapes

    @property
    def dtype(self):
        dtype = self.layer.attr["dtype"].type
        if dtype not in self.dtype_map:
            raise Exception("Dtype[{}] not in dtype_map".format(dtype))
        return self.dtype_map[dtype]

    @property
    def value(self):
        assert self.layer_type == "Const", "Only Const node has value."

        attr = self.layer.attr['value']
        field = getattr(attr, attr.WhichOneof('value'))
        return tensor_util.MakeNdarray(field)

    def get_attr(self, name):
        if name not in self.layer.attr:
            return None
        attr = self.layer.attr[name]
        field = attr.WhichOneof('value')
        value = getattr(attr, field) if field else None

        if isinstance(value, attr_value_pb2.AttrValue.ListValue):
            result = list(value.ListFields()[0][1])
            for i in range(len(result)):
                if isinstance(result[i], int):
                    result[i] = int(result[i])
                try:
                    if isinstance(result[i], long):
                        result[i] = int(result[i])
                except:
                    pass
            return result
        else:
            return value


class TFGraph(Graph):
    def __init__(self, model):
        super(TFGraph, self).__init__(model)

    def build(self):
        for layer in self.model.node:
            self.node_map[layer.name] = TFGraphNode(layer)

        for layer_name, node in self.node_map.items():
            for in_node in node.layer.input:
                if in_node not in self.node_map:
                    if in_node.strip().split(':')[0] in self.node_map:
                        self.connect(in_node.strip().split(':')[0], layer_name)
                    else:
                        raise Exception(
                            'input[{}] of node[{}] does not exist in node_map'.
                            format(in_node, layer_name))
                else:
                    self.connect(in_node, layer_name)

        super(TFGraph, self).build()


class TFParser(object):
    def __init__(self, pb_model, in_nodes=None, out_nodes=None, in_shapes=None):
        assert in_nodes is not None, "in_nodes should not be None"
        assert out_nodes is not None, "out_nodes should not be None"
        assert in_shapes is not None, "in_shapes should not be None"
        assert len(in_shapes) == len(
            in_nodes), "length of in_shapes and in_nodes should be equal"

        sess = tf.Session()
        with gfile.FastGFile(pb_model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        sess.run(tf.global_variables_initializer())

        self.tf_graph = TFGraph(sess.graph._as_graph_def(add_shapes=True)[0])
        self.tf_graph.build()
