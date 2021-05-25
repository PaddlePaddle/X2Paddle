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
from x2paddle.decoder.onnx_shape_inference import SymbolicShapeInference
from onnx.checker import ValidationError
from onnx.checker import check_model
from onnx.utils import polish_model
from onnx import helper, shape_inference
from onnx.helper import get_attribute_value, make_attribute
from onnx.shape_inference import infer_shapes
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx.numpy_helper import to_array
from onnx import AttributeProto, TensorProto, GraphProto
from collections import OrderedDict as Dict
import onnx
from onnx.helper import ValueInfoProto
import numpy as np
from copy import deepcopy
import logging as _logging
import os
import copy

default_op_domain = 'ai.onnx'
_logger = _logging.getLogger(__name__)


class ONNXGraphNode(GraphNode):
    def __init__(self, layer, layer_name=None):
        if layer_name is None:
            super(ONNXGraphNode, self).__init__(layer, layer.name)
        else:
            super(ONNXGraphNode, self).__init__(layer, layer_name)
        self.layer_type = layer.op_type
        self.attr_map = self.get_attr_map()
        self.out_shapes = list()
        self.dtype = None
        self.which_child = {}

    def get_attr_map(self):
        """
        convert ONNX node attributes to dict
        """
        return {
            attr.name: self.get_attribute_value(attr)
            for attr in self.layer.attribute
        }

    @property
    def value(self):
        assert 'Constant' in self.layer_type, "Only Constant | ConstantOfShape node has value."
        if 'value' not in self.attr_map:
            return None
        return self.attr_map['value']

    @property
    def name(self):
        if hasattr(self, 'index'):
            return "{}_p{}".format(self.layer_name, self.index)
        return self.layer_name

    def get_attribute_value(self, attr):
        """
        get_attribute_value enhanced
        """
        if attr.type == onnx.AttributeProto.TENSOR:
            dtype = np.dtype(TENSOR_TYPE_TO_NP_TYPE[attr.t.data_type])
            data = attr.t.raw_data
            value = np.frombuffer(
                data, dtype=dtype, count=(len(data) // dtype.itemsize))
        elif attr.type == onnx.AttributeProto.STRING:
            value = attr.s
            value = value.decode() if isinstance(value, bytes) else value
        else:
            value = get_attribute_value(attr)
        return value

    def get_attr(self, name, default=None):
        """
        get_attribute_value from attr_map
        """
        if name not in self.attr_map:
            return default
        return self.attr_map[name]

    def output(self, index=0):
        if index > 0 and len(self.layer.output) <= index:
            raise IndexError(
                'Output numbers of Node:{} is {} <= index:{}'.format(
                    self.layer_name, len(self.layer.output), index))
        return self.layer.output[index]


class ONNXGraphDataNode(GraphNode):
    def __init__(self, layer, layer_name=None, is_global_input=False):
        if layer_name is None:
            super(ONNXGraphDataNode, self).__init__(layer, layer.name)
        else:
            super(ONNXGraphDataNode, self).__init__(layer, layer_name)
        if is_global_input:
            self.layer_type = 'place_holder'
        else:
            self.layer_type = 'create_parameter'
        self.layer_name = layer_name
        self.weight = None
        self.embeded_as = None
        self.which_child = {}

    @property
    def out_shapes(self):
        if isinstance(self.layer, ValueInfoProto):
            values = self.layer.type.tensor_type.shape.dim
            out_shapes = list()
            shape = list()
            for dim in values:
                if dim.dim_value == 0:
                    shape.append(-1)
                else:
                    shape.append(dim.dim_value)
            out_shapes.append(shape)
            return out_shapes
        elif isinstance(self.layer, TensorProto):
            values = self.layer.dims
            out_shapes = list()
            shape = list()
            for dim in values:
                if dim == 0:
                    shape.append(-1)
                else:
                    shape.append(dim)
            out_shapes.append(shape)
            return out_shapes
        else:
            values = self.layer.dims
            out_shapes = list()
            out_shapes.append(values)
            return out_shapes

    @property
    def name(self):
        return self.layer_name

    @property
    def dtype(self):
        if isinstance(self.layer, ValueInfoProto):
            dtype = self.layer.type.tensor_type.elem_type
            return TENSOR_TYPE_TO_NP_TYPE[dtype]
        else:
            dtype = self.layer.data_type
            return TENSOR_TYPE_TO_NP_TYPE[dtype]


class ONNXGraph(Graph):
    def __init__(self, onnx_model):
        super(ONNXGraph, self).__init__(onnx_model)
        self.fixed_input_shape = {}
        self.initializer = {}
        self.place_holder_nodes = list()
        self.value_infos = {}
        self.graph = onnx_model.graph
        self.get_place_holder_nodes()
        print("shape inferencing ...")
        self.graph = SymbolicShapeInference.infer_shapes(
            onnx_model, fixed_input_shape=self.fixed_input_shape)
        if self.graph is None:
            print('[WARNING] Shape inference by ONNX offical interface.')
            onnx_model = shape_inference.infer_shapes(onnx_model)
            self.graph = onnx_model.graph
        print("shape inferenced.")
        self.build()
        self.collect_value_infos()
        self.allocate_shapes()
        self.graph_name = "ONNXModel"

    def get_inner_nodes(self):
        """
        generate inner node of ONNX model
        """
        inner_nodes = []
        if not isinstance(self.graph, onnx.GraphProto):
            logger.error('graph is not a GraphProto instance')
            return
        for initializer in self.graph.initializer:
            name = initializer.name
            inner_nodes.append(name)
        return inner_nodes

    def get_symbolic_shape(self, dims):
        shape = []
        for dim in dims:
            if dim.HasField('dim_param'):
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        return shape

    def check_input_shape(self, vi):
        if vi.type.HasField('tensor_type'):
            for dim in vi.type.tensor_type.shape.dim:
                if dim.HasField(
                        'dim_param') and vi.name not in self.fixed_input_shape:
                    shape = self.get_symbolic_shape(
                        vi.type.tensor_type.shape.dim)
                    print(
                        "Unknown shape for input tensor[tensor name: '{}'] -> shape: {}, Please define shape of input here,\nNote:you can use visualization tools like Netron to check input shape."
                        .format(vi.name, shape))
                    right_shape_been_input = False
                    while not right_shape_been_input:
                        try:
                            shape = raw_input(
                                "Shape of Input(e.g. -1,3,224,224), enter 'N' to skip: "
                            )
                        except NameError:
                            shape = input(
                                "Shape of Input(e.g. -1,3,224,224), enter 'N' to skip: "
                            )
                        if shape.count("-1") > 1:
                            print("Only 1 dimension can be -1, type again:)")
                        else:
                            right_shape_been_input = True
                    if shape == 'N':
                        break
                    shape = [int(dim) for dim in shape.strip().split(',')]
                    assert shape.count(-1) <= 1, "Only one dimension can be -1"
                    self.fixed_input_shape[vi.name] = shape
                    break

    def get_place_holder_nodes(self):
        """
        generate place_holder node of ONNX model
        """
        inner_nodes = self.get_inner_nodes()
        for ipt_vi in self.graph.input:
            if ipt_vi.name not in inner_nodes:
                self.check_input_shape(ipt_vi)
                self.place_holder_nodes.append(ipt_vi.name)

    def get_output_nodes(self):
        """
        generate output_nodes node of ONNX model
        """
        self.output_nodes = [value.name for value in self.graph.output]

    def is_place_holder_nodes(self, layer):
        """
        return layer is or not place_holder node
        """
        if layer in self.place_holder_nodes:
            return True
        return False

    def build(self):
        """
        build topo_sort of ONNX model
        """
        for layer in self.graph.node:
            node = ONNXGraphNode(layer)
            self.node_map[layer.name] = node

        for layer in self.graph.input:
            if layer.name not in self.node_map:
                is_place_holder = self.is_place_holder_nodes(layer.name)
                self.node_map[layer.name] = ONNXGraphDataNode(
                    layer,
                    layer_name=layer.name,
                    is_global_input=is_place_holder)

        #set data node's weight
        for initializer in self.graph.initializer:
            name = initializer.name
            weight = to_array(initializer)
            if name in self.node_map:
                if isinstance(self.node_map[name], ONNXGraphDataNode):
                    self.node_map[name].weight = weight
                    self.node_map[name].embeded_as = []
            else:
                self.node_map[name] = ONNXGraphDataNode(
                    initializer, layer_name=name, is_global_input=False)
                self.node_map[name].weight = weight
                self.node_map[name].embeded_as = []

        #generate connection between nodes for topo
        for layer_name, node in self.node_map.items():
            if isinstance(node, ONNXGraphNode):
                self.build_connection(layer_name, node)

        #generate topo
        super(ONNXGraph, self).build()

        self.input_nodes = copy.deepcopy(self.place_holder_nodes)

    def build_connection(self, layer_name, node):
        """
        find connection for nodes
        """
        for idx, in_node in enumerate(node.layer.input):
            if in_node == '':
                continue
            if in_node not in self.node_map:
                flag = 0
                for nd in self.graph.node:
                    for idx, opt in enumerate(nd.output):
                        if opt == in_node:
                            self.connect(nd.name, layer_name)
                            flag = 1
                            if nd.name in node.which_child:
                                for n_i, n_ipt in enumerate(node.inputs):
                                    if first_i == n_i:
                                        continue
                                    if n_ipt == nd.name:
                                        new_nd_name = "{}/{}".format(nd.name,
                                                                     n_i)
                                        if new_nd_name not in node.which_child:
                                            node.which_child[new_nd_name] = idx
                                            break
                            else:
                                first_i = node.inputs.index(nd.name)
                                node.which_child[nd.name] = idx
                            self.node_map[nd.name].index = 0
                            break
                    if flag == 1:
                        break
                if flag == 0:
                    raise Exception(
                        'input[{}] of node[{}] does not exist in node_map'.
                        format(in_node, layer_name))
            else:
                self.connect(in_node, layer_name)

    def get_input_node(self, node, idx=0, copy=False):
        if len(node.which_child) == 0:
            ipt_node = super(ONNXGraph, self).get_node(node.inputs[idx], copy)
            return ipt_node
        else:
            ipt_node = super(ONNXGraph, self).get_node(node.inputs[idx], copy)
            new_ipt_name = "{}/{}".format(ipt_node.layer_name, idx)
            if new_ipt_name in node.which_child:
                ipt_node.index = node.which_child[new_ipt_name]
            else:
                if ipt_node.layer_name in node.which_child:
                    ipt_node.index = node.which_child[ipt_node.layer_name]

            return ipt_node

    def graph_weights(self):
        """
        generator for weights
        """

        if not isinstance(self.graph, onnx.GraphProto):
            logger.error('graph is not a GraphProto instance')
            return

        for initializer in self.graph.initializer:
            name = initializer.name
            weight = to_array(initializer)
            yield name, weight

    def collect_value_infos(self):
        """
        collect value/type info for an ONNX model
        """
        assert isinstance(self.graph,
                          onnx.GraphProto), 'model is not a ModelProto instance'

        for item in self.graph.value_info:
            self.value_infos[item.name] = {
                'dtype':
                TENSOR_TYPE_TO_NP_TYPE[item.type.tensor_type.elem_type],
                'shape':
                [dim.dim_value for dim in item.type.tensor_type.shape.dim],
                'external': False
            }

    def allocate_shapes(self):
        """
        run shape inference
        """
        for layer in self.graph.node:
            node = self.node_map[layer.name]
            for opt in layer.output:
                if opt in self.value_infos:
                    value_info = self.value_infos[opt]
                    #if len(value_info['shape']) == 0 or value_info[
                    #        'dtype'] is None or 0 in value_info['shape']:
                    #    #TODO add node shape inference
                    shape = value_info['shape']
                    for idx in range(len(shape)):
                        if shape[idx] == 0:
                            shape[idx] = -1
                    node.out_shapes.append(shape)
                    node.dtype = value_info['dtype']
                else:
                    node.out_shapes.append([])


class ONNXDecoder(object):
    def __init__(self, onnx_model):
        onnx_model = onnx.load(onnx_model)
        print('model ir_version: {}, op version: {}'.format(
            onnx_model.ir_version, onnx_model.opset_import[0].version))
        self.op_set = onnx_model.opset_import[0].version

        check_model(onnx_model)

        onnx_model = self.optimize_model_skip_op(onnx_model)
        onnx_model = self.optimize_node_name(onnx_model)
        self.graph = ONNXGraph(onnx_model)

    def build_value_refs(self, nodes):
        """
        build op reference of inputs and outputs
        """
        input_refs = Dict()
        output_refs = Dict()
        for idx, node in enumerate(nodes):
            for val_name in node.input:
                input_refs.setdefault(val_name, set()).add(idx)
            for val_name in node.output:
                output_refs.setdefault(val_name, set()).add(idx)
        return input_refs, output_refs

    def skip_node_forward(self, nodes, src_output_name, dst_input_name,
                          input_refs):
        """
        skip nodes between src_output_name -> dst_input_name and connect this pair
        """
        processed = 0
        for next_idx in input_refs[src_output_name]:
            next_node = nodes[next_idx]
            for val_idx, next_input_name in enumerate(next_node.input):
                if next_input_name == src_output_name:
                    next_node.input[val_idx] = dst_input_name
                    processed += 1
        return processed

    def skip_node_backward(self, nodes, src_input_name, dst_output_name,
                           output_refs):
        """
        skip nodes between dst_output_name -> src_input_name and connect this pair
        """
        processed = 0
        for prev_idx in output_refs[src_input_name]:
            prev_node = nodes[prev_idx]
            for val_idx, prev_output_name in enumerate(prev_node.output):
                if prev_output_name == src_input_name:
                    prev_node.output[val_idx] = dst_output_name
                    processed += 1
        return processed

    def optimize_model_skip_op(self, model, op_list=None):
        """
        skip ops can be bypassed for inference
        """
        nodes = model.graph.node
        if op_list is None:
            op_list = ['Dropout']
        input_refs, output_refs = self.build_value_refs(nodes)
        ret = type(model)()
        ret.CopyFrom(model)
        ret_nodes = ret.graph.node
        nodes_to_remove = []
        for node_idx, node in enumerate(nodes):
            if not (node.domain == default_op_domain or node.domain == ''):
                continue
            op_type = node.op_type
            if not (op_type in op_list):
                continue
            if op_type in ['Dropout']:
                input_name = node.input[0]
                output_name = node.output[0]
            elif not (len(node.input) == 1 and len(node.output) == 1):
                print(
                    'currently only 1-input-1-output op supported, skip required %d: %s',
                    node_idx, node.op_type)
                continue
            else:
                input_name = node.input[0]
                output_name = node.output[0]

            if output_name in input_refs:
                processed = self.skip_node_forward(ret_nodes, output_name,
                                                   input_name, input_refs)
            elif input_name in output_refs:
                processed = self.skip_node_backward(ret_nodes, input_name,
                                                    output_name, output_refs)
            else:
                processed = -1
            if processed > 0:
                nodes_to_remove.append(node_idx)
                for value_info in ret.graph.value_info:
                    for output in node.output:
                        if value_info.name == output:
                            ret.graph.value_info.remove(value_info)

                print('skip op {}: {} -> {} -> {}'.format(
                    node_idx, input_name, node.op_type, output_name))
            elif processed == 0:
                print('weird, no node processed')
            else:
                print('standalone op {}: {} -> {} -> {} not skipped'.format(
                    node_idx, input_name, node.op_type, output_name))

        nodes_to_remove.sort(reverse=True)
        for node_idx in nodes_to_remove:
            ret_nodes.pop(node_idx)
        return ret

    def optimize_model_strip_initializer(self, model, keep_input_only=True):
        """
        strip weights for inference
        """
        nodes = model.graph.node
        input_refs, output_refs = self.build_value_refs(nodes)
        out_names = [val.name for val in model.graph.output]

        ret = type(model)()
        ret.CopyFrom(model)
        # strip initializers
        ret.graph.ClearField('initializer')
        ret_initializers = ret.graph.initializer
        for initializer in model.graph.initializer:
            name = initializer.name
            if name in input_refs:
                ret_initializers.add().CopyFrom(initializer)
            elif not keep_input_only and name in output_refs:
                ret_initializers.add().CopyFrom(initializer)
            else:
                dtype = TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]

        # strip inputs
        ret.graph.ClearField('input')
        ret_inputs = ret.graph.input
        for item in model.graph.input:
            name = item.name
            if name in input_refs or name in out_names:
                ret_inputs.add().CopyFrom(item)
        return ret

    def make_variable_name(self, name):
        """
        make a valid code name for ParamAttr
        """
        if name == '':
            raise ValueError('name should not be empty')
        for s in ' .*?\\/-:':
            name = name.replace(s, '_')
        return 'x2paddle_' + name

    def optimize_node_name(self, model):
        """
        standardize variable name for paddle's code
        """
        graph = model.graph
        for initializer in graph.initializer:
            initializer.name = self.make_variable_name(initializer.name)
        for ipt in graph.input:
            ipt.name = self.make_variable_name(ipt.name)
        for output in graph.output:
            output.name = self.make_variable_name(output.name)
        for item in graph.value_info:
            item.name = self.make_variable_name(item.name)
        for node in graph.node:
            node.name = node.output[0]
            node.name = self.make_variable_name(node.name)
            for i in range(len(node.input)):
                if node.input[i] == '':
                    continue
                else:
                    node.input[i] = self.make_variable_name(node.input[i])
            for i in range(len(node.output)):
                node.output[i] = self.make_variable_name(node.output[i])
        return model
