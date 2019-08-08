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

from x2paddle.decoder.caffe_decoder import CaffeGraph
from x2paddle.core.util import *


class CaffeOptimizer(object):
    layers_with_act = ['Convolution', 'Deconvolution', 'InnerProduct']
    activation_ops = ['ReLU', 'Sigmoid']

    def __init__(self, mapper):
        self.graph = mapper.graph

    def merge_bn_scale(self):
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            if node.layer_type == 'Scale':
                parent_node = self.graph.get_bottom_node(node, idx=0)
                if parent_node.layer_type == 'BatchNorm':
                    is_delete_node = True if len(
                        parent_node.outputs) == 1 else False
                    parent_fluid_layer = parent_node.fluid_code.layers[0]
                    input = parent_fluid_layer.inputs
                    parent_param_attr = parent_fluid_layer.param_attr
                    parent_param_attr['param_attr'] = string(node.layer_name +
                                                             '_scale')
                    parent_param_attr['bias_attr'] = string(node.layer_name +
                                                            '_offset')
                    if is_delete_node:
                        parent_node.fluid_code.clear()
                    node.fluid_code.clear()
                    node.fluid_code.add_layer("batch_norm",
                                              inputs=input,
                                              output=node,
                                              param_attr=parent_param_attr)

    def merge_op_activation(self):
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            if node.layer_type in self.activation_ops:
                parent_node = self.graph.get_bottom_node(node, idx=0)
                if parent_node.layer_type in self.layers_with_act:
                    is_delete_node = True if len(
                        parent_node.outputs) == 1 else False
                    parent_fluid_layer = parent_node.fluid_code.layers[0]
                    input = parent_fluid_layer.inputs
                    parent_param_attr = parent_fluid_layer.param_attr
                    parent_param_attr['act'] = string(node.layer_type.lower())
                    op = parent_fluid_layer.op
                    if is_delete_node:
                        parent_node.fluid_code.clear()
                    node.fluid_code.clear()
                    node.fluid_code.add_layer(op,
                                              inputs=input,
                                              output=node,
                                              param_attr=parent_param_attr)
