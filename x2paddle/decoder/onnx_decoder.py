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
from onnx.checker import ValidationError
from onnx.checker import check_model
from onnx.utils import polish_model
from onnx.version_converter import convert_version
from onnx.helper import get_attribute_value, make_attribute
from onnx.shape_inference import infer_shapes
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx.numpy_helper import to_array
from collections import OrderedDict as Dict
import onnx
import copy
import numpy as np
from copy import deepcopy

DEFAULT_OP_DOMAIN = 'ai.onnx'


def skip_node_forward(nodes, src_output_name, dst_input_name, input_refs):
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


def skip_node_backward(nodes, src_input_name, dst_output_name, output_refs):
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

class ONNXGraphNode(GraphNode):
    def __init__(self, layer, layer_name=None):
        if layer_name is None:
            super(ONNXGraphNode, self).__init__(layer, layer.name)
        else:
            super(ONNXGraphNode, self).__init__(layer, layer_name)
        self.layer_type = layer.op_type
        self.fluid_code = FluidCode()
        self.attr_map = self.get_attr_map()
        self.dtype_map = {1: "float32", 3: "int32", 9: "int64"}
    def __eq__(self, other):
        if isinstance(other,str):
            if self.layer.name == other:
                return True
        elif self.layer.name == other.layer.name:
            return True
        return False
        
    def get_attr_map(self):
        return {attr.name: self.get_attribute_value2(attr) for attr in self.layer.attribute} 
        
    def get_attribute_value2(self,attr):
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

    def set_inputs(self):
        self.inputs = self.layer.input

    def set_outputs(self):
        self.outputs = self.layer.output

    def node_attrs(node):
        """
        convert ONNX node attributes to dict
        """
        return {attr.name: get_attribute_value2(attr) for attr in node.attribute}  # dict

    def get_attr(self, name, default=None):
        if name not in self.attr_map:
            return default
        return self.attr_map[name]

class ONNXGraph(Graph):
    def __init__(self, model):
        super(ONNXGraph, self).__init__(model)
        self.inner_nodes = []
        self.initializer = {}

    def get_inner_nodes(self):
        """
        generate inner node of ONNX model
        """
        if not isinstance(self.model, onnx.GraphProto):
            logger.error('graph is not a GraphProto instance')
            return
        for initializer in self.model.initializer:
            name = initializer.name
            self.inner_nodes.append(name)
    
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

    def _get_topo_sort(self):
        """
        generate topo_sort of ONNX model
        """
        node_in_degrees = [len(set(node.input)) for node in self.model.node]
        node_out_degrees = [len(set(node.output)) for node in self.model.node]
        input_refs, output_refs = self.build_value_refs(self.model.node)
        for val_name in input_refs:
            if val_name not in output_refs:
                for node_idx in input_refs[val_name]:
                    node_in_degrees[node_idx] -= 1
        queue = []
        for node_idx, degree in enumerate(node_in_degrees):
            if degree == 0:
                queue.append(node_idx)
        while len(queue) > 0:
            node_idx = queue.pop(0)
            self.topo_sort.append(self.model.node[node_idx].name)
            for val_name in self.model.node[node_idx].output:
                output_refs[val_name].remove(node_idx)
                if len(output_refs[val_name]) > 0:
                    continue
                output_refs.pop(val_name)
                if val_name not in input_refs:
                    continue
                for next_idx in input_refs[val_name]:
                    node_in_degrees[next_idx] -= 1
                    self.connect(self.model.node[node_idx].name, self.model.node[next_idx].name)
                    if node_in_degrees[next_idx] == 0:
                        queue.insert(0, next_idx)

    def build(self):
        """
        build topo_sort of ONNX model
        """ 
        for layer in self.model.node:
            self.node_map[layer.name] = ONNXGraphNode(layer)
        self.get_inner_nodes()
        self._get_topo_sort()
        
    def get_nodes(self, names, forGenCode=False, copy=False):
        nodes = []
        for name in names:
            nodes.add(self.get_node( name, forGenCode=forGenCode, copy=copy))
    
    def get_node(self, name, forGenCode=False, copy=False):
        """
        get ONNXGraphNode or (name of input/out) by name
        """
        if name not in self.node_map:
            if name.split(':')[0] in self.node_map:
                name_prefix, idx = name.split(':')
                if copy:
                    node = deepcopy(self.node_map[name_prefix])
                else:
                    node = self.node_map[name_prefix]
                node.index = int(idx)
                return node
            elif forGenCode:
                return name
            else:
                raise Exception("Graph doesn't have node [%s]." % name)
        else:
            if copy:
                node = deepcopy(self.node_map[name])
            else:
                node = self.node_map[name]
            return node

        
class ONNXDecoder(object):
    def __init__(self, onnx_model):
        model = onnx.load(onnx_model)
        check_model(model)
        model = polish_model(model)

        model = self.optimize_model_skip_op_for_inference(model)
        model = self.optimize_model_strip_initializer(model)
        
        self.model = model
        self.standardize_variable_name(model.graph)
        
        graph_def = model.graph
        self.onnx_graph = ONNXGraph(graph_def)
        self.onnx_graph.build()
        self.graph_value_infos = self.inferred_model_value_info(model)
        
        # add weight info
        for name, weight in self.graph_weights(graph_def):
            value_info = self.graph_value_infos[name]
            value_info['embeded_as'] = []
            value_info['get_weight'] = (lambda w: lambda: w.tolist())(
                weight)
            value_info['weight'] = weight

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
    
    def optimize_model_skip_op_for_inference(self, model, op_list=None):
        """
        skip ops can be bypassed for inference
        """
        if op_list is None:
            op_list = ['Dropout']
            
        nodes = model.graph.node
        input_refs, output_refs = self.build_value_refs(nodes)

        ret = type(model)()
        ret.CopyFrom(model)
        ret.graph.ClearField(
            'value_info')  # WORKAROUND: onnx do not drop old value_info
        ret_nodes = ret.graph.node
        nodes_to_remove = []
        for node_idx, node in enumerate(nodes):
            if not (node.domain == DEFAULT_OP_DOMAIN or node.domain == ''):
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
                processed = skip_node_forward(ret_nodes, output_name, input_name,
                                              input_refs)
            elif input_name in output_refs:
                processed = skip_node_backward(ret_nodes, input_name, output_name,
                                               output_refs)
            else:
                processed = -1

            if processed > 0:
                nodes_to_remove.append(node_idx)
                print('skip op %d: %s -> %s -> %s', node_idx, input_name,
                             node.op_type, output_name)
            elif processed == 0:
                print('weird, no node processed')
            else:
                print('standalone op %d: %s -> %s -> %s not skipped',
                               node_idx, input_name, node.op_type, output_name)

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
        ret.graph.ClearField(
            'value_info')  # WORKAROUND: onnx do not drop old value_info

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
        for s in ' .*?\\/-:':  #
            name = name.replace(s, '_')
        return '_'+ name
    
    def graph_weights(self, graph):
        """
        generator for weights of an ONNX model
        """

        if not isinstance(graph, onnx.GraphProto):
            logger.error('graph is not a GraphProto instance')
            return

        for initializer in graph.initializer:
            name = initializer.name
            weight = to_array(initializer)
            yield name, weight
            
    def standardize_variable_name(self, graph):
        """
        standardize variable name for paddle's code
        """
        
        for initializer in graph.initializer:
            initializer.name = self.make_variable_name(initializer.name)
        for ipt in graph.input:
            ipt.name = self.make_variable_name(ipt.name)
        for output in graph.output:
            output.name = self.make_variable_name(output.name)
        for item in graph.value_info: 
            item.name = self.make_variable_name(item.name)
        for node in graph.node:
            if node.name == '':
                node.name = node.output[0]
            node.name = self.make_variable_name(node.name)
            for i in range(len(node.input)):
                node.input[i]=self.make_variable_name(node.input[i])
            for i in range(len(node.output)):
                node.output[i]=self.make_variable_name(node.output[i])
            
    def tensor_dtype(self, tensor):
        """
        get ONNX tensor in np.dtype
        """

        return TENSOR_TYPE_TO_NP_TYPE[tensor.type.tensor_type.elem_type]

    def tensor_shape(self, tensor):
        """
        get ONNX tensor shape
        """

        return [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]
        
    def inferred_model_value_info(self, model):
        """
        collect value/type info for an ONNX model
        """
        model = infer_shapes(model)
        graph = model.graph
        value_info = Dict()
        for item in graph.value_info:
            value_info[item.name] = dict(
                dtype=self.tensor_dtype(item),
                shape=self.tensor_shape(item),
                external=False,
            )
        for item in graph.input:
            assert item.name not in value_info
            value_info[item.name] = dict(
                dtype=self.tensor_dtype(item),
                shape=self.tensor_shape(item),
                external=True,
            )
        for item in graph.output:
            #assert item.name not in value_info, 'bypass-model not supported'
            value_info[item.name] = dict(
                dtype=self.tensor_dtype(item),
                shape=self.tensor_shape(item),
                external=True,
            )
        return value_info

    def save_inference_model(self, save_dir):
        onnx.save(self.model, save_dir+'model.onnx')
        
    