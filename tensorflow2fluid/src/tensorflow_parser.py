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

import tensorflow
from tensorflow_graph import TensorflowGraph
from tensorflow.python.framework import tensor_util
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes


class TensorflowCkptParser(object):
    def __init__(self,
                 meta_file,
                 checkpoint_file,
                 dest_nodes,
                 input_shape=None,
                 in_nodes=None):
        graph_def = None
        self.weights = None
        with tensorflow.Session() as sess:
            if meta_file is None:
                raise Exception("meta_file must be provided")
            new_saver = tensorflow.train.import_meta_graph(meta_file)
            if checkpoint_file is not None:
                self.weights = dict()
                new_saver.restore(
                    sess, tensorflow.train.latest_checkpoint(checkpoint_file))
                for var in tensorflow.global_variables():
                    value = var.eval(sess)
                    self.weights[var.name.split(':')[0]] = value

            graph_def, ver = tensorflow.get_default_graph()._as_graph_def(
                add_shapes=True)

        if in_nodes is not None and input_shape is not None:
            graph_def = strip_unused_lib.strip_unused(
                input_graph_def=graph_def,
                input_node_names=in_nodes,
                output_node_names=dest_nodes,
                placeholder_type_enum=dtypes.float32.as_datatype_enum)

            self.tf_graph = TensorflowGraph(graph_def)
        else:
            raise Exception('in_nodes and output_nodes need be provided')
        self.tf_graph.build()


class TensorflowPbParser(object):
    def __init__(self, pb_file, dest_nodes, input_shape=None, in_nodes=None):
        with open(pb_file) as f:
            serialized = f.read()
        tensorflow.reset_default_graph()
        original_graph_def = tensorflow.GraphDef()
        original_graph_def.ParseFromString(serialized)
        original_graph_def = strip_unused_lib.strip_unused(
            input_graph_def=original_graph_def,
            input_node_names=in_nodes,
            output_node_names=dest_nodes,
            placeholder_type_enum=dtypes.float32.as_datatype_enum)

        graph_def = tensorflow.GraphDef()
        graph_def.ParseFromString(original_graph_def.SerializeToString())
        in_type_list = dict()
        for node in graph_def.node:
            if node.name in in_nodes:
                in_type_list[node.name] = node.attr['dtype'].type

        input_shape = list(input_shape)
        if not isinstance(input_shape[0], list):
            input_shape = [input_shape]

        input_map = dict()
        for i in range(len(input_shape)):
            if in_type_list[in_nodes[i]] == 1 or in_type_list[
                    in_nodes[i]] == 0:
                dtype = tensorflow.float32
                x = tensorflow.placeholder(dtype, shape=input_shape[i])
            elif in_type_list[in_nodes[i]] == 3:
                dtype = tensorflow.int32
                x = tensorflow.placehoder(dtype, shape=input_shape[i])
            else:
                raise Exception(
                    "Unexpected dtype for input, only support float32 and int32 now"
                )
            input_map[in_nodes[i] + ":0"] = x

        tensorflow.import_graph_def(graph_def, name="", input_map=input_map)
        graph_def = tensorflow.get_default_graph()._as_graph_def(
            add_shapes=True)[0]
        node = graph_def.node[0]
        self.tf_graph = TensorflowGraph(graph_def)
        self.tf_graph.build()

        self.weights = dict()
        for node in graph_def.node:
            if node.op.lower() == "const":
                try:
                    node.attr['value'].tensor.tensor_content
                    weight = tensor_util.MakeNdarray(node.attr['value'].tensor)
                    self.weights[node.name] = weight
                except:
                    continue
