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


class TensorflowParser(object):
    def __init__(self,
                 meta_file,
                 checkpoint_file,
                 dest_nodes,
                 input_shape=None,
                 in_nodes=None):
        graph_def = None
        self.weights = dict()
        with tensorflow.Session() as sess:
            if meta_file is None:
                raise Exception("meta_file must be provided")
            new_saver = tensorflow.train.import_meta_graph(meta_file)
            if checkpoint_file is not None:
                new_saver.restore(
                    sess, tensorflow.train.latest_checkpoint(checkpoint_file))
                for var in tensorflow.global_variables():
                    value = var.eval(sess)
                    self.weights[var.name.split(':')[0]] = value

            graph_def, ver = tensorflow.get_default_graph()._as_graph_def(
                add_shapes=True)

        if in_nodes is not None and input_shape is not None:
            from tensorflow.python.tools import strip_unused_lib
            from tensorflow.python.framework import dtypes
            graph_def = strip_unused_lib.strip_unused(
                input_graph_def=graph_def,
                input_node_names=in_nodes,
                output_node_names=dest_nodes,
                placeholder_type_enum=dtypes.float32.as_datatype_enum)

            input_list = [None]
            for i in range(len(input_shape)):
                input_list.append(tensorflow.Dimension(input_shape[i]))
            tensor_input = tensorflow.TensorShape(input_list)

            self.tf_graph = TensorflowGraph(graph_def)
            for node in self.tf_graph.model.node:
                if node.name in in_nodes:
                    node.attr['shape'].list.shape.extend(
                        [tensor_input.as_proto()])
                    node.attr['_output_shapes'].list.shape.pop()
                    node.attr['_output_shapes'].list.shape.extend(
                        [tensor_input.as_proto()])
        else:
            raise Exception('in_nodes and output_nodes need be provided')
        self.tf_graph.build()
