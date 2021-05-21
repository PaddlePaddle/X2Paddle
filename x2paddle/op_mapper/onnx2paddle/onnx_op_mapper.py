# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import sys
from x2paddle.op_mapper.onnx2paddle.opset9 import OpSet9
from x2paddle.decoder.onnx_decoder import ONNXGraphNode
from x2paddle.core.program import PaddleGraph


class ONNXOpMapper():
    def __init__(self, decoder):
        self.support_op_sets = [9, ]
        self.default_op_set = 9
        self.graph = decoder.graph
        self.paddle_graph = PaddleGraph(parent_layer=None, source_type="onnx")
        self.paddle_graph.outputs = self.graph.output_nodes
        self.opset = self.create_opset(decoder)
        if not self.op_checker():
            raise Exception("Model is not supported yet.")

        print("Total nodes: {}".format(
            sum([
                isinstance(node, ONNXGraphNode)
                for name, node in self.graph.node_map.items()
            ])))
        print("Nodes converting ...")
        for i, node_name in enumerate(self.graph.topo_sort):
            sys.stderr.write("\rConverting node {} ...     ".format(i + 1))
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if hasattr(self.opset, op):
                func = getattr(self.opset, op)
                func(node)
            elif op in self.opset.directly_map_ops:
                self.opset.directly_map(node)
            elif op in self.opset.elementwise_ops:
                self.opset.elementwise_map(node)
        print("\nNodes converted.")
        self.paddle_graph.set_name(self.graph.graph_name)
        self.paddle_graph.set_parameters(self.opset.weights)
        self.paddle_graph.set_inputs_info(self.opset.inputs_info)

    def op_checker(self):
        unsupported_ops = set()
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if not hasattr(self.opset, op) and \
                op not in self.opset.directly_map_ops and \
                op not in self.opset.elementwise_ops:
                unsupported_ops.add(op)
        if len(unsupported_ops) == 0:
            return True
        else:
            if len(unsupported_ops) > 0:
                print("\n========= {} OPs are not supported yet ===========".
                      format(len(unsupported_ops)))
            for op in unsupported_ops:
                print("========== {} ============".format(op))
            return False

    def create_opset(self, decoder):
        run_op_set = self.default_op_set
        opset = ''
        if decoder.op_set in self.support_op_sets:
            opset = 'OpSet' + str(decoder.op_set)
        elif decoder.op_set < self.default_op_set:
            opset = 'OpSet' + str(self.default_op_set)
        else:
            for op_set in self.support_op_sets:
                if decoder.op_set > op_set:
                    run_op_set = op_set
                else:
                    break
            opset = 'OpSet' + str(run_op_set)
        print(
            'Now, onnx2paddle support convert onnx model opset_verison {},'
            'opset_verison of your onnx model is {}, automatically treated as op_set: {}.'
            .format(self.support_op_sets, decoder.op_set, run_op_set))
        return eval(opset)(decoder, self.paddle_graph)
