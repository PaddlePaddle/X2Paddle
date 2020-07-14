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

from x2paddle.core.fluid_code import Layer
from x2paddle.decoder.pytorch_decoder import PyTorchGraph
from x2paddle.core.util import *

class PyTorchOptimizer(object):
    layers_with_act = ['conv2d', 'linear']
    activation_ops = ['relu']

    def __init__(self, mapper):
        self.graph = mapper.graph
        
    def merge_op_activation(self):
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            if node.layer_type in self.activation_ops:
                parent_node = self.graph.get_input_node(node, idx=0)
                if parent_node.layer_type in self.layers_with_act:
                    is_delete_node = True if len(
                        parent_node.outputs) == 1 else False
                    parent_fluid_layer = parent_node.fluid_code.layers[0]
                    parent_inputs = parent_fluid_layer.inputs
                    parent_param_attr = parent_fluid_layer.param_attr
                    parent_param_attr['act'] = string(node.layer_type.lower())
                    op = parent_fluid_layer.op
                    name = parent_fluid_layer.output
                    output = node.layer_name
                    if is_delete_node:
                        parent_node.fluid_code.clear()
                    node.fluid_code.clear()
                    node.fluid_code.add_dygraph(op,
                                           name=name,
                                           inputs=parent_inputs,
                                           output=node,
                                           param_attr=parent_param_attr
                                           )
                    
    def merge_if_condition(self):
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            if node.layer_type == 'control_if':
                parent_node = self.graph.get_input_node(node, idx=0)
                is_delete_node = True if len(
                        parent_node.outputs) == 1 else False
                parent_fluid_layer = parent_node.fluid_code.layers[-1]
                parent_fluid_layer_part = parent_fluid_layer.split('=')
                if_code = '='.join(parent_fluid_layer_part[1:])
                if is_delete_node:
                    parent_node.fluid_code.layers.pop()
                node.fluid_code.clear()
                node.fluid_code.add_note("if {}:".format(if_code))
                
    def remove_dropout(self):
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            if node.layer_type == 'dropout':
                parent_node = self.graph.get_input_node(node, idx=0)
                dropout_opt_name = node.fluid_code.layers[-1].split(' = ')[0]
                dropout_ipt_name = node.fluid_code.layers[-1].split('(')[-1][:-1]
                parent_node_layer = parent_node.fluid_code.layers[-1]
                if isinstance(parent_node_layer, Layer):
                    # 只能是静态图op的情况
                    parent_node.output = dropout_opt_name
                else:
                    # 动态图op最后一行为xx0 = self.xx1(xx2)
                    part = parent_node_layer.split('=')
                    part[0] = part[0].replace(dropout_ipt_name, dropout_opt_name)
                    parent_node.fluid_code.layers[-1] = '='.join(part)
                node.fluid_code.clear()
                    

                        
                
                        

