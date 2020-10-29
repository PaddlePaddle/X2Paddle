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

from x2paddle.op_mapper.onnx2paddle.opset9 import OpSet9, custom_layers
from x2paddle.core.op_mapper import OpMapper
from x2paddle.decoder.onnx_decoder import ONNXGraph, ONNXGraphNode, ONNXGraphDataNode


class ONNXOpMapper(OpMapper):
    def __init__(self, decoder):
        super(ONNXOpMapper, self).__init__()
        self.support_op_sets = [9, ]
        self.default_op_set = 9
        self.graph = decoder.graph
        self.opset = self.create_opset(decoder)
        if not self.op_checker():
            raise Exception("Model are not supported yet.")
        #mapping op
        print("Total nodes: {}".format(
            sum([
                isinstance(node, ONNXGraphNode)
                for name, node in self.graph.node_map.items()
            ])))

        print("Nodes converting ...")
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if hasattr(self.opset, op):
                func = getattr(self.opset, op)
                func(node)
            elif op in self.opset.default_op_mapping:
                self.opset.directly_map(node)
            elif op in custom_layers:
                self.opset.deal_custom_layer(node)
            elif op in self.opset.elementwise_ops:
                self.opset.elementwise_map(node)
        print("Nodes converted.")
        self.weights = self.opset.weights
        self.omit_nodes = self.opset.omit_nodes
        self.used_custom_layers = self.opset.used_custom_layers

    def op_checker(self):
        unsupported_ops = set()
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if not hasattr(self.opset, op) and \
                op not in self.opset.default_op_mapping and \
                op not in custom_layers and \
                op not in self.opset.elementwise_ops:
                unsupported_ops.add(op)
        if len(unsupported_ops) == 0:
            return True
        else:
            print("There are {} ops not supported yet, list as below".format(
                len(unsupported_ops)))
            for op in unsupported_ops:
                print(op)
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
        return eval(opset)(decoder)
