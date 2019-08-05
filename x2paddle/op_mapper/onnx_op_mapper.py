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

from x2paddle.decoder.onnx_decoder import ONNXGraph
from x2paddle.core.graph import GraphNode
from x2paddle.core.op_mapper import OpMapper
from x2paddle.core.util import *
from x2paddle.core.fluid_code import Layer
from x2paddle.core.fluid_code import FluidCode
import numpy as np
from tool.onnx_to_paddle import get_dynamic_shape
import logging as _logging
from collections import OrderedDict as _dict
_logger = _logging.getLogger(__name__)

def _shape_or_none(value_infos, val_name):
    if isinstance(val_name, GraphNode):
        val_name = val_name.layer_name
    if val_name not in value_infos:
        return None
    value_info = value_infos[val_name]
    if 'shape' not in value_info:
        return None
    return list(value_info['shape'])
    
def _shape(value_infos, val_name):
    if isinstance(val_name, GraphNode):
        val_name = val_name.layer_name
    return list(value_infos[val_name]['shape'])
    
def _const_weight_or_none(value_infos, val_name):
    if isinstance(val_name, GraphNode):
        val_name = val_name.layer_name
    if val_name not in value_infos:
        return None
    value_info = value_infos[val_name]
    const_value = value_info.get('const_value', None)
    if const_value:
        return const_value
    get_weight_func = value_info.get('get_weight', None)
    if get_weight_func:
        return get_weight_func()
    return None

def _dtype_or_none(value_infos, val_name):
    if isinstance(val_name, GraphNode):
        val_name = val_name.layer_name
    if val_name not in value_infos:
        return None
    value_info = value_infos[val_name]
    if 'dtype' not in value_info:
        return None
    return np.dtype(value_info['dtype'])

DEFAULT_OP_MAPPING_FIELD_VALUES = _dict()
DEFAULT_OP_MAPPING_FIELD_VALUES['FLUID_OP'] = ''
DEFAULT_OP_MAPPING_FIELD_VALUES['FLUID_INPUT_ARGS'] = None
DEFAULT_OP_MAPPING_FIELD_VALUES['FLUID_OUTPUT_ARGS'] = None
DEFAULT_OP_MAPPING_FIELD_VALUES['ATTR_MAPPING'] = dict(
)  # dict(onnx_attr_from=fluid_attr_to)
DEFAULT_OP_MAPPING_FIELD_VALUES['DEFAULTS'] = dict()  # dict(fluid_attr=default)
DEFAULT_OP_MAPPING_FIELD_VALUES[
    'INPUT_PERM'] = None  # sampler: [idx_onnx_arg...]
DEFAULT_OP_MAPPING_FIELD_VALUES[
    'OUTPUT_PERM'] = None  # sampler: [idx_onnx_arg...]
DEFAULT_OP_MAPPING_FIELD_VALUES['FILL_NAME_FIELD'] = True

DEFAULT_OP_MAPPING = {
        ## nil ops ##
        'RandomUniform':['uniform_random', [], ['Out'], dict(high='max', low='min'), dict(), None, None, False],
        'RandomNormal':['gaussian_random', [], ['Out'], dict(scale='std'),dict(), None, None, False],
        ## unary ops ##
        'Abs': ['abs', ['X'], ['Out']],
        'ArgMax': ['argmax', ['X'], ['Out'], dict(keepdims='')],
        'ArgMin': ['argmin', ['X'], ['Out'], dict(keepdims='')],
        'Ceil': ['ceil', ['X'], ['Out']],
        'Clip': ['clip', ['X'], ['Out']], # attrs bypassed
        'Cos': ['cos', ['X'], ['Out']],
        'Elu': ['elu', ['X'], ['Out']],
        'Exp': ['exp', ['X'], ['Out']],
        'Flatten': ['flatten', ['X'], ['Out']], # attrs bypassed, FIXME: emit flatten2
        'Floor': ['floor', ['X'], ['Out']],
        'Gather': ['gather', ['X'], ['Out'], dict(axis='')],
        'LeakyRelu': ['leaky_relu', ['X'], ['Out']],
        'Log': ['log', ['X'], ['Out']],
        'Reciprocal': ['reciprocal', ['X'], ['Out']],
        'Relu': ['relu', ['X'], ['Out']],
        'Selu': ['selu', ['X'], ['Out'], dict(gamma='scale')],
        'Shape': ['shape', ['X'], ['Out']], # FIXME: out is int64 vs int32
        'Shrink': ['softshrink', ['X'], ['Out'], dict(bias='', labmd='')],
        'Sigmoid': ['sigmoid', ['X'], ['Out']],
        'Sin': ['sin', ['X'], ['Out']],
        'Squeeze': ['squeeze', ['X'], ['Out']], # attrs bypassed, FIXME: emit squeeze2
        'Softplus': ['softplus', ['X'], ['Out']],
        # FIXME: default axis = -1, reshape required before and after
        'Softmax': ['softmax', ['X'], ['Out'], dict(axis='')],
        'Softsign': ['softsign', ['X'], ['Out']],
        'Sqrt': ['sqrt', ['X'], ['Out']],
        'Tanh': ['tanh', ['X'], ['Out']],
        'ThresholdedRelu': ['thresholded_relu', ['X'], ['Out'], dict(alpha='threshold')],
        'Unsqueeze': ['unsqueeze', ['X'], ['Out']], # attrs bypassed, FIXME: emit unsqueeze2
        ## binary ops ##
        'Add': ['elementwise_add', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        'And': ['logical_and', ['X', 'Y'], ['Out']],
        'Div': ['elementwise_div', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        'Equal': ['equal', ['X', 'Y'], ['Out'], dict(), dict(), None, None, False],
        'Greater': ['less_than', ['X', 'Y'], ['Out'], dict(), dict(), [1, 0], None, False],
        'Less': ['less_than', ['X', 'Y'], ['Out'], dict(), dict(), None, None, False],
        'MatMul': ['matmul', ['X', 'Y'], ['Out']], # defaults excluded for transpose_x vs transpose_X
        'Max': ['elementwise_max', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        'Min': ['elementwise_min', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        'Mul': ['elementwise_mul', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        'Not': ['logical_not', ['X', 'Y'], ['Out']],
        'OneHot': ['one_hot', ['Input', 'Depth'], ['Out'], dict(axis=''), dict(),[0, 1], None, False], # assuming values=[0, 1], axis=-1 and drop them
        'Or': ['logical_or', ['X', 'Y'], ['Out']],
        'Pow': ['elementwise_pow', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)], # TODO: pow for scalar exponent
        'Sub': ['elementwise_sub', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        'Xor': ['logical_xor', ['X', 'Y'], ['Out']],
        # reduce ops
        'ReduceMax': ['reduce_max', ['X'], ['Out'], dict(axes='dim', keepdims='keep_dim')],
        'ReduceMean': ['reduce_mean', ['X'], ['Out'], dict(axes='dim', keepdims='keep_dim')],
        'ReduceMin': ['reduce_min', ['X'], ['Out'], dict(axes='dim', keepdims='keep_dim')],
        'ReduceProd': ['reduce_prod', ['X'], ['Out'], dict(axes='dim', keepdims='keep_dim')],
        'ReduceSum': ['reduce_sum', ['X'], ['Out'], dict(axes='dim', keepdims='keep_dim')],
        # other ops
        'Scatter': ['scatter', ['X', 'Index', 'Updates'], ['Out']],
        'TopK': ['topk', ['X', 'K'], ['Out', 'Indices']],
        'Expand': ['expand', ['X'], ['Out'], dict(shape='expand_times')],
}

DEFAULT_IOA_CONSTRAINT = {
    'ArgMax': [
        (lambda i, o, a: a.get('keepdims', 1) == 1,
         'only keepdims = 0 is supported'),
    ],
    'ArgMin': [
        (lambda i, o, a: a.get('keepdims', 1) == 1,
         'only keepdims = 0 is supported'),
    ],
    'Gather': [
        (lambda i, o, a: a.get('axis', 0) == 0, 'only axis = 0 is supported'),
    ],
    'Shrink': [
        (lambda i, o, a: a.get('bias', 0) == a.get('lambd', 0.5),
         'only SoftShrink with bias = lambd is supported'),
    ],
    #        'Softmax':
    #            [(lambda i, o, a: a.get('axis', 1) == -2, 'Paddle fluid Softmax works on dim -2 only'),
    #            ],
    'OneHot': [
        (lambda i, o, a: a.get('axis', -1) == -1,
         'only axis = -1 is supported'),
    ],
    'Scatter': [
        (lambda i, o, a: a.get('axis', 0) == 0, 'only axis = 0 is supported'),
    ],
    'TopK': [
        (lambda i, o, a: a.get('axis', -1) == -1,
         'only axis = -1 is supported'),
    ],
}

class ONNXOpMapper(OpMapper):
    def __init__(self, decoder):
        super(ONNXOpMapper, self).__init__()
        self.decoder = decoder
        self.graph = decoder.onnx_graph
        self.input_shapes = []
        self.weights = dict()
        self.omit_weights = list()
        self.omit_nodes = list()
        self.external_inputs = []
        self.external_inputs =self.get_external_inputs()
        
    def op_checker(self):
        unsupported_ops = set()
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if not hasattr(self, op) and op not in DEFAULT_OP_MAPPING:
                unsupported_ops.add(op)
        if len(unsupported_ops) == 0:
            return True
        else:
            print("There are {} ops not supported yet, list as below".format(
                len(unsupported_ops)))
            for op in unsupported_ops:
                print(op)
            return False

    def run(self):
        print("Total nodes: {}".format(len(self.graph.topo_sort)))
        # check if ops in model are all supported
        if not self.op_checker():
            raise Exception("Model are not supported yet.")

            
        #generate code for input data
        for name in self.external_inputs:
            value_info = self.decoder.graph_value_infos[name]
            self.input_shapes.append(value_info['shape'])
            attr = {
            "dtype": string(value_info['dtype']),
            "shape": value_info['shape'],
            "name": string(name),
            "append_batch_size":'False'}
            fluid_code = FluidCode()
            fluid_code.add_layer("data",
                                  inputs=None,
                                  output=name,
                                  param_attr=attr)
            self.net_code += fluid_code.gen_codes()
            
        #mapping op
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            print('Translating node{}: op is {}'.format(node.layer_name, op))
            if op in DEFAULT_OP_MAPPING:
                 self._default(node)
            elif hasattr(self, op):
                func = getattr(self, op)
                func(node)
                
        #mapping weight info
        for name, value_info in self.decoder.graph_value_infos.items():
            if 'weight' in value_info and name not in self.omit_weights:
                weight = value_info['weight']
                self.weights[name] = weight
                if name not in self.omit_nodes:
                    shape = weight.shape
                    dtype = weight.dtype
                    if shape is None:
                        shape = _shape_or_none(value_infos, name)
                    if shape is None:
                        shape = list(value.shape)
                    fluid_code = FluidCode()
                    attr = {
                            'dtype': string(dtype),
                            'shape': shape,
                            'name':string(name),
                            'default_initializer':'Constant(0.0)',
                            'attr':string(name)}
                    fluid_code.add_layer("create_parameter",
                                      inputs=None,
                                      output=name,
                                      param_attr=attr)
                    self.net_code += fluid_code.gen_codes()

        #generate code for op
        for i in range(len(self.graph.topo_sort)):
            if name not in self.omit_nodes:
                node_name = self.graph.topo_sort[i]
                if node_name in self.omit_nodes:
                    continue
                node = self.graph.get_node(node_name)
                self.net_code += node.fluid_code.gen_codes()


    def get_external_inputs(self):
        external_inputs = []
        input_nodes = [value.name for value in self.graph.model.input]
        for node in input_nodes:
            if node not in self.graph.inner_nodes:
                external_inputs.append(node)
        return external_inputs
    
    def _default(self, node, *args, name='', **kwargs):
        inputs = node.layer.input
        outputs = node.layer.output
        op_type = node.layer_type
        attrs = node.attr_map
        
        info = DEFAULT_OP_MAPPING[op_type]
        info.extend(list(DEFAULT_OP_MAPPING_FIELD_VALUES.values())[len(info):])
        (
            fluid_op,
            fluid_input_args,
            fluid_output_args,
            attr_mapping,
            default_attrs,
            input_perm,
            output_perm,
            fill_name_field,
        ) = info

        if fluid_op in DEFAULT_IOA_CONSTRAINT:
            for predicate, message in DEFAULT_IOA_CONSTRAINT[fluid_op]:
                assert predicate(inputs, outputs, attrs), message

        # bypass if key absent, drop if mapped key is '' or '_'
        mapped_attrs = {
            attr_mapping.get(key, key): value
            for key, value in attrs.items()
        }
        if '' in mapped_attrs:
            mapped_attrs.pop('')
        if '_' in mapped_attrs:
            mapped_attrs.pop('_')
        fluid_attrs = default_attrs.copy()
        fluid_attrs.update(mapped_attrs)  # as new attrs

        val_inps = inputs if input_perm is None else list(map(lambda i: inputs[i],
                                                         input_perm))
        val_outs = outputs if output_perm is None else list(map(lambda i: outputs[i],
                                                           output_perm))
        attr = fluid_attrs
        if fluid_op not in ['shape','gather']:
            attr['name'] = string(node.layer_name)
        node.fluid_code.add_layer(fluid_op, 
                                inputs=', '.join(val_inps), output = val_outs[0], param_attr=attr)

    def _pad_if_asymmetric(self, node, pads, val_name, value_infos):  # pads: SSEE
        assert len(pads) & 1 == 0
        symmetric = True
        ndims = len(pads) // 2
        for idx_dim in range(ndims):
            if pads[idx_dim] != pads[ndims + idx_dim]:
                symmetric = False
                break
        if symmetric:
                return pads[:ndims], val_name

        val_padded = self.Pad(node, op_independent=False)
        
        return [0] * ndims, val_padded
    
    def Pad(self, node, op_independent=True):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        pads = node.get_attr('pads') # required
        mode = node.get_attr('mode', 'constant')  # optional
        value = node.get_attr('value', 0.)  # optional
        data_shape = _shape_or_none(self.decoder.graph_value_infos, val_x)
        output_shape = _shape_or_none(self.decoder.graph_value_infos, node)
        assume_pad2d = False
        attr = {}
        if len(pads) == 4:
            assume_pad2d |= mode != 'constant'
            if data_shape:
                assume_pad2d |= data_shape and len(data_shape) == 4  # NCHW
            if output_shape:
                assume_pad2d |= output_shape and len(output_shape) == 4  # NCHW
        if assume_pad2d:
            fluid_op = 'pad2d'
            attr['data_format'] = string('NCHW')
            attr['mode'] = string(mode)
        else:
            attr = {'pad_value': value}
            assert mode == 'constant', 'mode {} is supported only in pad2d'.format(mode)
            fluid_op = 'pad'
        if len(pads) == 4:
            paddings = np.array(pads).reshape(
                (-1, 2)).transpose().flatten().tolist()  # SSEE -> SESE
        elif len(pads) == 8:
            paddings = np.array(pads).reshape(
            (-1, 4)).transpose().flatten().tolist()  # SSEE -> SESE
        attr['paddings'] = paddings
        
        if op_independent:

            attr['name'] = string(node.layer_name)
            node.fluid_code.add_layer(fluid_op, 
                                    inputs=val_x, output=node, param_attr=attr)
        else:
            attr['name'] = string(node.layer_name+'_paded')
            node.fluid_code.add_layer(fluid_op, 
                                    inputs=val_x, output=node.layer_name+'_paded', param_attr=attr)
            return node.layer_name+'_paded'
        
#     def Unsqueeze(self, node):
#         val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
#         axes = node.get_attr('axes')
#         if isinstance(val_x, str):   
#             if val_x  in self.decoder.graph_value_infos:
#                 self.omit_nodes.append(val_x)
#                 self.omit_nodes.append(node.layer_name)
#                 value_info = self.decoder.graph_value_infos[val_x]
#                 self.decoder.graph_value_infos[node.layer_name] = self.decoder.graph_value_infos.pop(val_x)
#                 weight = self.decoder.graph_value_infos[node.layer_name]['weight']
#                 for idx, axs in enumerate(axes):
#                     weight = np.expand_dims(weight, axis=idx+axs)
#                 self.decoder.graph_value_infos[node.layer_name]['weight'] = weight
            
#         attr = {
#                 'shape': shape,
#                 'name': string(node.layer_name)
#             }
#         node.fluid_code.add_layer('reshape', 
#                                 inputs=val_x, output=node, param_attr=attr)
    def Constant(self, node):
        val_output = self.graph.get_node(node.layer.output[0], forGenCode=True, copy=True)

        # interpretation
        value = node.get_attr('value') # required
        dtype = np.dtype(value.dtype)
        output_dtype = _dtype_or_none(self.decoder.graph_value_infos, val_output)
        if output_dtype:
            assert dtype == output_dtype, 'tensor dtype unmatches storage dtype'

    #    dtype = np.dtype('float32') # HINT: force to float32
        shape = node.get_attr('shape', None)  #
        if shape is None:
            shape = _shape_or_none(self.decoder.graph_value_infos, val_output)
        if shape is None:
            shape = list(value.shape)
            _logger.warning(
                'in (Constant -> %s): '
                'attribute "shape" of %s not inferred, '
                'using value as 1-D tensor may lead to fails', outputs, val_output)

        # generation
        value = value.tolist()
        if len(value) == 1:  # scalar
            shape = [1]  # WORKAROUND: bad scalar support
            value = value[0]
            if  dtype.name == 'int64':
                dtype = 'int32'
            attr= {
                    'shape':shape,
                    'dtype':string(dtype),
                    'value':value
                }
            node.fluid_code.add_layer('fill_constant', 
                                    inputs=None, output=node, param_attr=attr)
        self.decoder.graph_value_infos[val_output.layer_name]['const_value'] = value

    def Resize(self, node):
        # I/O
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        val_scales = self.graph.get_node(node.layer.input[1], forGenCode=True, copy=True)
        val_y, = self.graph.get_node(node.layer.output[0], forGenCode=True, copy=True)
        
        # interpretation
        # output shape
        out_shape_ = _shape_or_none(self.decoder.graph_value_infos, val_y)
        if out_shape_ is not None:
            assert len(out_shape_) == 4, 'only 4-D Tensor as X and Y supported'
            out_shape_ = out_shape_[2:]
        # try scales
        scales = _const_weight_or_none(self.decoder.graph_value_infos, val_scales)
        if scales is not None:
            assert len(scales) == 4, 'only 4-D Tensor as X and Y supported'
            assert scales[0] == 1 and scales[
                1] == 1, 'only scale on (NC)HW supported'
            assert scales[2] == scales[
                3], 'only aspect-ratio-invariant scale supported'
        scale = scales[2] if scales else None
        # try input shape
        if scale is None:
            assert out_shape_, 'neither scales nor output shape is available'
            out_shape = out_shape_
        else:
            out_shape = None
            if out_shape_ is None:
                in_shape = _shape_or_none(self.decoder.graph_value_infos, val_x)
                assert in_shape is not None, 'out_shape required but not inferrable'
                assert len(in_shape) == 4, 'only 4-D Tensor as X and Y supported'
                out_shape_ = [in_shape[2] * scale, in_shape[3] * scale]
        
        mode = node.get_attr('mode', 'nearest')
        fluid_op = 'resize_{}'.format(mode)  # not sure bilinear will be linear?
        name_attr = ', name={}'.format(repr(name)) if name else ''
        
        attr = {
                'scale':scale,
                'out_shape':out_shape,
                'name':string(node.layer_name)
                }
        
        # generation
        node.fluid_code.add_layer(fluid_op, inputs=val_x, output = node, param_attr=attr)

    def ConstantOfShape(self, node):
        """
        onnx::ConstantOfShape-9:
        """
        # I/O
        val_shape = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)

        shape = _const_weight_or_none(self.decoder.graph_value_infos, val_shape)
        if shape is None:
            shape = _shape_or_none(self.decoder.graph_value_infos, node)
        assert shape is not None, (
            'given shape is neither const value nor deductible from output, '
            'this is not supported')
        
        value = node.get_attr('value') # required
        dtype = value.dtype
        # generation
        value = value.tolist()
        if len(value) == 1:  # scalar
            shape = [1]  # WORKAROUND: bad scalar support
            value = value[0]
            if  dtype.name == 'int64':
                dtype = 'int32'
            attr= {
                    'shape':shape,
                    'dtype':string(dtype),
                    'value':value
                }
            node.fluid_code.add_layer('fill_constant', 
                                    inputs=None, output=node, param_attr=attr)

        self.decoder.graph_value_infos[node.layer_name]['const_value'] = value

    def Split(self, node):
        """
        onnx::Split-2:
        """
        # I/O
        val_input = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        var_outs = [val for val in node.layer.input]

        # interpretation
        fluid_op = 'split'
        split = node.get_attr['split']  # required
        axis =  node.get_attr('axis', 0)  # optional
        attr= {
                'split':split,
                'axis':axis,
                'name':string(node.layer_name)
                }
        # generation
        node.fluid_code.add_layer('split', inputs=val_input, output = var_outs, param_attr=attr)

    def Reshape(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        val_shape = self.graph.get_node(node.layer.input[1], forGenCode=True, copy=True)
        val_reshaped = self.graph.get_node(node.layer.output[0], forGenCode=True, copy=True)
        
        var_shape = val_shape if isinstance(val_shape, str) else val_shape.layer_name
        var_reshaped = val_reshaped if isinstance(val_reshaped, str) else val_reshaped.layer_name
        
        if isinstance(val_shape, GraphNode):
            shape = get_dynamic_shape(self.decoder.model, var_shape, self.input_shapes)
        else:
            shape = _const_weight_or_none(self.decoder.graph_value_infos, var_shape)

            is_const_shape = shape and 'const_value' in self.decoder.graph_value_infos[var_shape]

            if shape is None:
                shape = _shape_or_none(self.decoder.graph_value_infos, var_reshaped)

            if shape is None:
                shape = [1, -1]  # who knows
                _logger.warning(
                    'in %s(%s -> Reshape -> %s): '
                    'input "shape" not inferred, use [1, -1] as dummy value, '
                    'the behavior of Paddle fluid maybe undefined', name, inputs,
                    outputs)
        
        # if input reshape is initializer, reshape initializer by numpy
        if isinstance(val_x, str):
            if val_x  in self.decoder.graph_value_infos:
                self.omit_nodes.append(val_x)
                self.omit_weights.append(val_shape)
                self.omit_nodes.append(node.layer_name)
                value_info = self.decoder.graph_value_infos[val_x]
                self.decoder.graph_value_infos[node.layer_name] = self.decoder.graph_value_infos.pop(val_x)
                self.decoder.graph_value_infos[node.layer_name]['weight'] = self.decoder.graph_value_infos[node.layer_name]['weight'].reshape(shape)

        attr = {
                'shape': shape,
                'name': string(node.layer_name)}
#         if is_const_shape:
#             node.fluid_code.add_layer('reshape', 
#                                 inputs=val_x, output=node, param_attr=attr)
#         else:
#             var_shape_int32 = var_shape + '_int32'  # explicit variable
#             node.fluid_code.add_layer('cast', 
#                                 inputs=val_shape, output=var_shape_int32, param_attr=attr)
        node.fluid_code.add_layer('reshape', 
                                inputs=val_x, output=node, param_attr=attr)

    def Cast(self, node):
        val_input = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        val_output = self.graph.get_node(node.layer.output[0], forGenCode=True, copy=True)

        # interpretation
        dtype = node.get_attr('to') # required
        if not isinstance(dtype, np.dtype):  # additional: possible np.dtype
            dtype = TENSOR_TYPE_TO_NP_TYPE[dtype]
        output_dtype = _dtype_or_none(self.decoder.graph_value_infos, val_output)
        if output_dtype:
            assert dtype == output_dtype, 'dtype of to unmatches output'
        attr={
            'dtype':string(dtype)
        }
        node.fluid_code.add_layer('cast', 
                                inputs=val_input, output=node, param_attr=attr)

    def AveragePool(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        assert node.get_attr(
                'auto_pad',
                'NOTSET') == 'NOTSET', 'only auto_pad = NOTSET is supported'
        kernel_shape = node.get_attr("kernel_shape")
        poolnd = len(kernel_shape)
        strides = node.get_attr("strides")
        pad_mode = node.get_attr("pads")
        ceil_mode = bool(node.get_attr('ceil_mode', 0))
        pads = node.get_attr('pads', [0] * (poolnd * 2))
        fluid_op = 'pool{}d'.format(poolnd)
        assert 2 <= poolnd <= 3, 'only pool2d and pool3d is supported'
        
        paddings, val_x = self._pad_if_asymmetric(node, pads, val_x, self.decoder.graph_value_infos)
        attr = {
            "pool_size": kernel_shape,
            "pool_type": string('avg'),
            "pool_stride": strides,
            "pool_padding":paddings,
            "ceil_mode":ceil_mode,
            "exclusive":'True',
            "name":string(node.layer_name)
        }
        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)
        
    def _roi_pool(self, node, fluid_op=None):
        # I/O
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        val_rois = self.graph.get_node(node.layer.input[1], forGenCode=True, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], forGenCode=True, copy=True)

        # interpretation
        spatial_scale = node.get_attr('spatial_scale') # required
        pooled_height, pooled_width = node.get_attr('pooled_shape')  # required
        
        attr ={'pooled_height':pooled_height,
              'spatial_scale':spatial_scale}
        feature_attr = ''
        is_max_pool = fluid_op == 'roi_pool'
        if 'sampling_ratio' in node.attr_map:  #
            sampling_ratio = node.get_attr['sampling_ratio']
            attr['sampling_ratio']= sampling_ratio
        if 'output_channels' in node.attr_map:  #
            output_channels = node.get_attr['output_channels']
            attr['output_channels']= output_channels
            
        # generation
        node.fluid_code.add_layer(fluid_op,
                                  inputs=','.join([valx,val_rois]),
                                  output=node,
                                  param_attr=attr)
    
    def RoiAlign(self, node):
        self._roi_pool(node, fluid_op='roi_align')
        
    def NonMaxSuppression(self, node):
        (val_boxes, val_scores, val_max_output_boxes_per_class, 
         val_iou_threshold, val_score_threshold)= self.graph.get_nodes(node.layer.input, forGenCode=True, copy=True)
        
        center_point_box = node.get_attr('center_point_box', 0)
        
        scores = _const_weight_or_none(self.decoder.graph_value_infos, val_scores)
        max_output_boxes_per_class = _const_weight_or_none(self.decoder.graph_value_infos, val_max_output_boxes_per_class)
        iou_threshold = _const_weight_or_none(self.decoder.graph_value_infos, val_iou_threshold)
        score_threshold = _const_weight_or_none(self.decoder.graph_value_infos, val_score_threshold)
        if center_point_box == 1:
            pass
                                
        attr={
            'scores':scores,
            'score_threshold':score_threshold,
            'nms_threshold':iou_threshold,
            'nms_top_k':max_output_boxes_per_class,
        }

    def Concat(self, node):
        inputs=[]
        for i in range(len(node.layer.input)):
            ipt = self.graph.get_node(node.layer.input[i], forGenCode=True, copy=True)
            if isinstance(ipt,str):
                inputs.append(ipt)
            else:
                inputs.append(ipt.layer_name)
        axis = node.get_attr('axis')
        attr={
            'axis': axis
        }
        node.fluid_code.add_layer('concat',
                                  inputs='[' + ', '.join(inputs) + ']',
                                  output=node,
                                  param_attr=attr)

    def Flatten(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        axis = node.get_attr('axis',1)
        attr = {
            "axis": str(axis),
            "name": string(node.layer_name)
        }
        node.fluid_code.add_layer('flatten',
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Gemm(self, node):
        val_a = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        val_b = self.graph.get_node(node.layer.input[1], forGenCode=True, copy=True)
        val_c = self.graph.get_node(node.layer.input[2], forGenCode=True, copy=True)
        
        alpha = node.get_attr('alpha', 1.)  # optional
        beta = node.get_attr('beta', 1.)  # optional
        trans_a = bool(node.get_attr('transA', 0))  # optional
        trans_b = bool(node.get_attr('transB', 0))  # optional
        val_mm = node.layer_name + '_mm'
        matmul_inputs={"x":val_a, "y":val_b}
        attr_matmul={
                    "transpose_x":trans_a,
                    "transpose_y":trans_b,
                    "alpha":alpha,
                    "name": string(val_mm)}
        node.fluid_code.add_layer('matmul',
                                  inputs=matmul_inputs,
                                  output=val_mm,
                                  param_attr=attr_matmul)
        if beta != 0:
            if beta == 1.:
                add_inputs = {"x": val_mm, "y": val_c}
                attr = {"name": string(node.layer_name)}
                node.fluid_code.add_layer("elementwise_add",
                                          inputs=add_inputs,
                                          output=node,
                                          param_attr=attr)
            else:
                pass

    def Add(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        val_x = self.graph.get_node(node.layer.input[1], forGenCode=True, copy=True)
        inputs = {"x": val_x,
                "y": val_y,
                 }
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer("elementwise_add",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)
    def Sum(self, node):
        var_inps = [val for val in node.layer.input]
        node.fluid_code.add_layer("sum",
                                  inputs='[' + ', '.join(var_inps) + ']',
                                  output=node)
    def MatMul(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        val_y = self.graph.get_node(node.layer.input[1], forGenCode=True, copy=True)
        inputs = {"x": val_x,
                "y": val_y}
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer("matmul",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)
    def LRN(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        size = node.get_attr('size')# required
        alpha = node.get_attr('alpha', 0.0001) # optional
        beta = node.get_attr('beta', 0.75) # optional
        bias = node.get_attr('bias', 1.0) # optional
        attr = {
            "n":max(1,size),
            "k":bias,
            "alpha":alpha,
            'beta':beta,
            "name": string(node.layer_name)
        }
        node.fluid_code.add_layer("lrn",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)
        
    def BatchNormalization(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        val_scale = self.graph.get_node(node.layer.input[1], forGenCode=True, copy=True)
        val_b = self.graph.get_node(node.layer.input[2], forGenCode=True, copy=True)
        val_mean = self.graph.get_node(node.layer.input[3], forGenCode=True, copy=True)
        val_var = self.graph.get_node(node.layer.input[4], forGenCode=True, copy=True)
        
        self.omit_nodes.append(val_scale)
        self.omit_nodes.append(val_b)
        self.omit_nodes.append(val_mean)
        self.omit_nodes.append(val_var)
        
        momentum = node.get_attr('momentum', .9)
        epsilon = node.get_attr('epsilon', 1e-5)
        
        attr = {
                "momentum":momentum,
                "epsilon":epsilon,
                "data_layout":string('NCHW'),
                "is_test":'True',
                "param_attr": string(val_scale),
                "bias_attr": string(val_b),
                "moving_mean_name": string(val_mean), 
                "moving_variance_name": string(val_var),
                "name": string(node.layer_name)}
        node.fluid_code.add_layer("batch_norm",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)
 
    def Softmax(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer("softmax",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def create_parameter(self, node, parameter):
        dtype = None

        value_infos = self.decoder.graph_value_infos[parameter]
#             weight = value_info['get_weight']
        shape = value_infos['shape']
        dtype = value_infos['dtype']
        if shape is None:
            shape = _shape_or_none(self.decoder.graph_value_infos, parameter)
        if shape is None:
            shape = list(value.shape)
        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(parameter),
            'attr': string(parameter),
        }
        node.fluid_code.add_layer("create_parameter",
                                  inputs=None,
                                  output=parameter,
                                  param_attr=attr)

    def Transpose(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        perm = node.get_attr('perm')
        attr = {'perm': perm,
                "name": string(node.layer_name)}
        node.fluid_code.add_layer("transpose",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Div(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        val_y = self.graph.get_node(node.layer.input[1], forGenCode=True, copy=True)
        inputs = {'x': val_x, 'y': val_y}
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer("elementwise_div",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)

    def Relu(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer("relu",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)
    def PRelu(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        val_slope = self.graph.get_node(node.layer.input[1], forGenCode=True, copy=True)
        attr = {
                "name": string(node.layer_name),
                "mode":string('channel')}

        if isinstance(val_slope,str): 
            attr["param_attr"] = string(val_slope)
        else:
            attr["param_attr"] = string(val_slope.layer_name)
        node.fluid_code.add_layer("prelu",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Squeeze(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        squeeze_dims = node.get_attr('squeeze_dims')
        attr = {'axes': squeeze_dims,
                "name": string(node.layer_name)}
        node.fluid_code.add_layer("squeeze",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Identity(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        node.fluid_code.add_layer("assign",
                                  inputs=val_x,
                                  output=node)

    def MaxPool(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        assert node.get_attr(
                'auto_pad',
                'NOTSET') == 'NOTSET', 'only auto_pad = NOTSET is supported'  # optional
        
        assert node.get_attr(
                "dilations") is None, 'only dilations = 0 is supported'  # optional
        
        kernel_shape = node.get_attr("kernel_shape")
        poolnd = len(kernel_shape)
        strides = node.get_attr("strides")
        pad_mode = node.get_attr("pads")
        ceil_mode = bool(node.get_attr('ceil_mode', 0))  # optional
        pads = node.get_attr('pads', [0] * (poolnd * 2))  # optional
        fluid_op = 'pool{}d'.format(poolnd)
        assert 2 <= poolnd <= 3, 'only pool2d and pool3d is supported'
        paddings, val_x = self._pad_if_asymmetric(node, pads, val_x, self.decoder.graph_value_infos)
        attr = {
            "pool_size": kernel_shape,
            "pool_type": string("max"),
            "pool_stride": strides,
            "pool_padding":paddings,
            "ceil_mode":ceil_mode,
            "name": string(node.layer_name),
            "exclusive":False
        }
        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)
    def GlobalAveragePool(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], forGenCode=True, copy=True)
        input_shape = _shape_or_none(self.decoder.graph_value_infos, val_x.layer_name)
        output_shape = _shape_or_none(self.decoder.graph_value_infos, val_y.layer_name)
        assert input_shape is not None or output_shape is not None, 'poolnd not inferred'  # N
        if input_shape:
            poolnd = len(input_shape) - 2  # NC...
        elif output_shape:
            poolnd = len(output_shape) - 2  # NC...
        assert 2 <= poolnd <= 3, 'only pool2d and pool3d is supported'
        fluid_op = 'pool{}d'.format(poolnd)
        attr = {
            "pool_type": string("avg"),
            "global_pooling":True,
            "name": string(node.layer_name)
        }
        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Conv(self, node):
        val_x = self.graph.get_node(node.layer.input[0], forGenCode=True, copy=True)
        val_w = self.graph.get_node(node.layer.input[1], forGenCode=True, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], forGenCode=True, copy=True)
        self.omit_nodes.append(val_w)

        input_shape = _shape_or_none(self.decoder.graph_value_infos, val_x if isinstance(val_x,str) else val_x.layer_name)

        has_bias = len(node.layer.input) == 3
        if has_bias:
            val_b = self.graph.get_node(node.layer.input[2], forGenCode=True, copy=True)
            self.omit_nodes.append(val_b)
            
        auto_pad = node.get_attr('auto_pad', 'NOTSET')

        kernel_shape = _shape(self.decoder.graph_value_infos, val_w)[2:]  # OI...
        assert kernel_shape == node.get_attr(
            'kernel_shape'), 'kernel_shape in attr unmatches value_info'  # HW
        convnd = len(kernel_shape)
        assert 2 <= convnd <= 3, 'only conv2d and conv3d is supported'
        num_out_channels = _shape(self.decoder.graph_value_infos, val_w)[0]  # OI...
        fluid_op = 'conv{}d'.format(convnd)

        num_groups = node.get_attr('group', 1) 
        strides = node.get_attr('strides', [1] * convnd)  # optional
        dilations = node.get_attr('dilations', [1] * convnd)  # optional
        pads = node.get_attr('pads', [0] * (convnd*2))  # optional
        
        paddings, val_x = self._pad_if_asymmetric(node, pads, val_x, self.decoder.graph_value_infos)
        
        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_UPPER":
            pad_h = get_same_padding(input_shape[2], kernel_shape[0], strides[0])
            pad_w = get_same_padding(input_shape[3], kernel_shape[1], strides[1])
            attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}

        attr = {
                "num_filters":num_out_channels,
                "filter_size":kernel_shape,
                "stride":strides,
                "padding":paddings,
                "dilation":dilations,
                "groups":num_groups,
                'param_attr':string(val_w),
                "name": string(node.layer_name)
            }
        if has_bias:
            attr["bias_attr"] = string(val_b)
        else:
            attr["bias_attr"] = False
        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)
    
