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

import functools
import numbers
import numpy as np
import math
import paddle.fluid as fluid
from x2paddle.decoder.pytorch_decoder import PyTorchGraph
from x2paddle.core.op_mapper import OpMapper
from x2paddle.core.util import *


class PyTorchOpMapper(OpMapper):
    def __init__(self, decoder):
        super(PyTorchOpMapper, self).__init__()
        self.graph = decoder.pytorch_graph
        self.weights = dict()
        self.datas_name = []

        print("Total nodes: {}".format(len(self.graph.topo_sort)))
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if hasattr(self, op):
                func = getattr(self, op)
                func(node)
            else:
                raise Exception(
                    "The op {} in model is not supported yet.".format(op))

    def op_checker(self):
        unsupported_ops = set()
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type.replace('::', '_')
            if not hasattr(self, op):
                unsupported_ops.add(op)
        if len(unsupported_ops) == 0:
            return True
        else:
            print("There are {} ops not supported yet, list as below".format(
                len(unsupported_ops)))
            for op in unsupported_ops:
                print(op)
            return False
        
    def get_input_name(self, node):
        if hasattr(node, "index"):
            return node.layer_name + "[{}]".format(node.index)
        else:
            return node.layer_name
        
    def data(self, node):
        node.fluid_code.add_note("{} = fluid.dygraph.base.to_variable({})".format(node.layer_name, node.layer_name))
        self.datas_name.append(node.layer_name) 
                
    def conv2d(self, node):
        input_node = self.graph.get_input_node(node, idx=0)
        node_attrs = node.attrs
        weight_name = node_attrs[1]
        bias_name = node_attrs[2]
        weights = node.params[weight_name].numpy()
        bias = node.params[bias_name].numpy()
        attrs = {}
        attrs['num_channels'] = weights.shape[1]
        attrs['num_filters'] = weights.shape[0]
        attrs['filter_size'] = weights.shape[2:]
        attrs['stride'] = node_attrs[3]
        attrs['padding'] = node_attrs[4]
        attrs['dilation'] = node_attrs[5]
        attrs['groups'] = node_attrs[6]
        attrs['param_attr'] = string(weight_name)
        attrs['bias_attr'] = string(bias_name)
        op_name = 'conv2d_' + node.layer_name
        self.weights[op_name + '.weight'] = node.params[weight_name].numpy()
        self.weights[op_name + '.bias'] = node.params[bias_name].numpy()
        node.fluid_code.add_dygraph("Conv2D",
                                    name = op_name,
                                    inputs=[input_node],
                                    output=node,
                                    param_attr=attrs)

    def flatten(self, node):
        input_node = self.graph.get_input_node(node, idx=0)
        node_attrs = node.attrs
        node.fluid_code.add_layer('shape',
                                  inputs=input_node,
                                  output=node.layer_name + '__shape',
                                  param_attr=None)
        node.fluid_code.add_note("{} = list({}.numpy())".format(node.layer_name + '__shape', node.layer_name + '__shape'))
        start_dim = node_attrs[1]
        end_dim = node_attrs[2] 
        if start_dim == 0 and end_dim == -1:
            node.fluid_code.add_note("{} = [1, functools.reduce(lambda x,y:x * y, {})]".format(node.layer_name + '__new_shape', 
                                                                                               node.layer_name + '__shape'))
        else:
            if end_dim == -1:
                end_dim = ''
            else:
                end_dim += 1        
            node.fluid_code.add_note("{} = {}[: {}]".format(node.layer_name + '__new_shape1', 
                                                            node.layer_name + '__shape', start_dim))
            node.fluid_code.add_note("{} = {}[{}: {}]".format(node.layer_name + '__new_shape2', 
                                                              node.layer_name + '__shape', 
                                                              start_dim, 
                                                              end_dim))
            if end_dim == '':
                node.fluid_code.add_note("{} = []".format(node.layer_name + '__new_shape3'))
            else:
                node.fluid_code.add_note("{} = {}[{}: ]".format(node.layer_name + '__new_shape3', 
                                                                node.layer_name + '__shape', 
                                                                end_dim))
            node.fluid_code.add_note("{} = [functools.reduce(lambda x,y:x * y, {})]".format(node.layer_name + '__new_shape2', 
                                                                                            node.layer_name + '__new_shape2'))
            node.fluid_code.add_note("{} = {} + {} + {}".format(node.layer_name + '__new_shape',
                                                                node.layer_name + '__new_shape1',
                                                                node.layer_name + '__new_shape2',
                                                                node.layer_name + '__new_shape3'))
        input_node_name = self.get_input_name(input_node)
        node.fluid_code.add_note("{} = fluid.layers.reshape({}, shape={})".format(node.layer_name, 
                                                                                  input_node_name, 
                                                                                  node.layer_name + '__new_shape'))
                                     
        
        

    def torch_relu(self, node):
        input_node = self.graph.get_input_node(node, idx=0)
        # inplace这个参数在paddle中未实现
        node.fluid_code.add_layer('relu',
                                  inputs=input_node,
                                  output=node.layer_name,
                                  param_attr=None)
        
    def torch_max_pool2d(self, node):
        input_node = self.graph.get_input_node(node, idx=0)
        node_attrs = node.attrs
        attrs= {}
        attrs['pool_size'] = node_attrs[1]
        attrs['pool_type'] = string('max')
        attrs['pool_stride'] = node_attrs[2]
        attrs['pool_padding'] = node_attrs[3]
        attrs['ceil_mode'] = node_attrs[5]
        # dilation这个空洞池化参数paddle中未实现
        assert node_attrs[4][0] == 1 and node_attrs[4][1] ==1, \
            "The dilation in MaxPool2d must be 1." 
        op_name = 'pool2d_' + node.layer_name
        node.fluid_code.add_dygraph("Pool2D",
                                    name=op_name,
                                    inputs=[input_node],
                                    output=node,
                                    param_attr=attrs)
        
        
    def torch_linear(self, node):
        input_node = self.graph.get_input_node(node, idx=0)
        node_attrs = node.attrs
        weight_name = node_attrs[1]
        bias_name = node_attrs[2]
        weights = node.params[weight_name].numpy()
        bias = node.params[bias_name].numpy()
        attrs = {}
        attrs['input_dim'] = weights.shape[1]
        attrs['output_dim'] = weights.shape[0]
        attrs['param_attr'] = string(weight_name)
        attrs['bias_attr'] = string(bias_name)
        op_name = 'linear_' + node.layer_name
        self.weights[op_name + '.weight'] = node.params[weight_name].numpy().transpose((1,0))
        self.weights[op_name + '.bias'] = node.params[bias_name].numpy()        
        node.fluid_code.add_dygraph("Linear",
                                    name=op_name,
                                    inputs=[input_node],
                                    output=node,
                                    param_attr=attrs)
        
    def torch_adaptive_avg_pool2d(self, node):
        input_node = self.graph.get_input_node(node, idx=0)
        node_attrs = node.attrs
        output_size = node_attrs[1]
        if isinstance(output_size, list):
            output_h = output_size[0]
            output_w = output_size[1]
        else:
            output_h = output_size
            output_w = output_size
        node.fluid_code.add_layer('shape',
                                  inputs=input_node,
                                  output=node.layer_name + '__shape',
                                  param_attr=None)
        node.fluid_code.add_note("{} = {}.numpy()".format(node.layer_name + '__shape', node.layer_name + '__shape'))
        node.fluid_code.add_note("{} = math.floor({}[3] / float({}))".format(node.layer_name + '__stride_w', 
                                                                             node.layer_name + '__shape',
                                                                             output_w))
        node.fluid_code.add_note("{} = math.floor({}[2] / float({}))".format(node.layer_name + '__stride_h', 
                                                                             node.layer_name + '__shape',
                                                                             output_h))
        node.fluid_code.add_note("{} = {}[3] - ({} - 1) * {}".format(node.layer_name + '__kernel_w',
                                                                     node.layer_name + '__shape',
                                                                     output_w,
                                                                     node.layer_name + '__stride_w'))
        node.fluid_code.add_note("{} = {}[2] - ({} - 1) * {}".format(node.layer_name + '__kernel_h',
                                                                     node.layer_name + '__shape',
                                                                     output_w,
                                                                     node.layer_name + '__stride_h'))
        op_name = 'pool2d_' + node.layer_name
        node.fluid_code.add_note("{} = fluid.dygraph.Pool2D(pool_size=[{}, {}], pool_type='max', pool_stride=[{}, {}], pool_padding=0)".format(
            op_name,
            node.layer_name + '__kernel_h', 
            node.layer_name + '__kernel_w',
            node.layer_name + '__stride_h', 
            node.layer_name + '__stride_w'))
        input_node_name = self.get_input_name(input_node)
        node.fluid_code.add_note("{} = {}({})".format(node.layer_name, op_name, input_node_name))
        
    def torch_dropout(self, node):
        input_node = self.graph.get_input_node(node, idx=0)
        op_name = 'dropout_' + node.layer_name
        node.fluid_code.add_dygraph("Dropout",
                                    name=op_name,
                                    inputs=[input_node],
                                    output=node,
                                    param_attr={'p': 0.0})
        
        

        
        
            
        
        
        
        
        
        