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

from x2paddle.core.graph import GraphNode
from x2paddle.core.op_mapper import OpMapper
from x2paddle.core.fluid_code import Layer
from x2paddle.core.fluid_code import FluidCode
from x2paddle.decoder.onnx_decoder import ONNXGraph, ONNXGraphNode, ONNXGraphDataNode
from x2paddle.op_mapper.onnx_directly_map import default_op_mapping_field_values
from x2paddle.op_mapper.onnx_directly_map import default_op_mapping
from x2paddle.op_mapper.onnx_directly_map import default_ioa_constraint
from x2paddle.op_mapper.onnx_custom_layer import *
from x2paddle.core.util import string
import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
import logging as _logging
from collections import OrderedDict as _dict
import math
import os
import shutil
from functools import reduce
import onnxruntime as rt
_logger = _logging.getLogger(__name__)


def _const_weight_or_none(node):
    if 'Constant' in node.layer_type:
        return node.value
    if isinstance(node, ONNXGraphDataNode):
        return node.weight
    return None


def get_same_padding(in_size, kernel_size, stride):
    new_size = int(math.ceil(in_size * 1.0 / stride))
    pad_size = (new_size - 1) * stride + kernel_size - in_size
    pad0 = int(pad_size / 2)
    pad1 = pad_size - pad0
    return [pad0, pad1]


class ONNXOpMapper(OpMapper):
    elementwise_ops = {
        'Add': 'elementwise_add',
        'Div': 'elementwise_div',
        'Sub': 'elementwise_sub',
        'Mul': 'elementwise_mul',
        'Pow': 'elementwise_pow',
    }

    def __init__(self, decoder, save_dir):
        super(ONNXOpMapper, self).__init__()
        self.decoder = decoder
        self.graph = decoder.onnx_graph
        self.input_shapes = []
        self.weights = dict()
        self.omit_nodes = list()
        self.used_custom_layers = dict()
        self.is_inference = False
        self.tmp_data_dir = os.path.join(save_dir, 'tmp_data')
        self.tmp_outputs_dict = {}
        self.get_output_shapes()

        if not self.op_checker():
            raise Exception("Model are not supported yet.")

        #mapping op
        print("Total nodes: {}".format(
            sum([
                isinstance(node, ONNXGraphNode)
                for name, node in self.graph.node_map.items()
            ])))
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if hasattr(self, op):
                func = getattr(self, op)
                func(node)
            elif op in default_op_mapping:
                self.directly_map(node)
            elif op in custom_layers:
                self.deal_custom_layer(node)
            elif op in self.elementwise_ops:
                self.elementwise_map(node)

        self.remove_tmp_data()

    def op_checker(self):
        unsupported_ops = set()
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if not hasattr(self, op) and \
                op not in default_op_mapping and \
                op not in custom_layers and \
                op not in self.elementwise_ops:
                unsupported_ops.add(op)
        if len(unsupported_ops) == 0:
            return True
        else:
            print("There are {} ops not supported yet, list as below".format(
                len(unsupported_ops)))
            for op in unsupported_ops:
                print(op)
            return False

    def get_results_of_inference(self, model, value_infos, data_nodes):
        if not os.path.exists(self.tmp_data_dir):
            os.makedirs(self.tmp_data_dir)
        inputs_dict = {}
        for data_node in data_nodes:
            value_info = value_infos[data_node]
            shape = value_info['shape']
            for i, dim_shape in enumerate(shape):
                if dim_shape == 0 and i == 0:
                    shape[i] = 1
                if dim_shape == 0 and i != 0:
                    assert 'shape of input is not assigned'
            ipt = np.random.random(shape).astype(value_info['dtype'])
            inputs_dict[data_node] = ipt

        model = onnx.shape_inference.infer_shapes(model)
        outputs = []

        for value_info in model.graph.value_info:
            outputs.append(value_info.name)

        model.graph.ClearField('output')
        model.graph.output.MergeFrom(model.graph.value_info)
        onnx.save(model, os.path.join(self.tmp_data_dir,
                                      'onnx_model_infer.onnx'))
        sess = rt.InferenceSession(
            os.path.join(self.tmp_data_dir, 'onnx_model_infer.onnx'))
        res = sess.run(None, input_feed=inputs_dict)
        self.tmp_outputs_dict = dict(zip(outputs, res))

        return

    def get_dynamic_shape(self, layer):
        """
        get dynamic shape from infer_result
        """
        if layer not in self.tmp_outputs_dict:
            return [None, None, None]
        output = self.tmp_outputs_dict[layer]
        return output.tolist(), output.dtype, output.shape

    def get_output_shapes(self):
        """
        build topo_sort of ONNX model
        """
        nodes = self.decoder.model.graph.node
        node_map = self.decoder.onnx_graph.node_map
        value_infos = self.decoder.onnx_graph.value_infos
        onnx_model = self.decoder.model
        for layer in nodes:
            node = node_map[layer.name]
            for opt in layer.output:
                if opt in value_infos:
                    value_info = value_infos[opt]
                    if len(value_info['shape']) == 0 or value_info[
                            'dtype'] is None or 0 in value_info['shape']:
                        if self.is_inference == False:
                            self.get_results_of_inference(
                                onnx_model, value_infos,
                                self.decoder.onnx_graph.place_holder_nodes)
                            self.is_inference = True
                        _, dtype, shape = self.get_dynamic_shape(opt)
                        node.out_shapes.append(shape)
                        node.dtype = dtype
                    else:
                        node.dtype = value_info['dtype']
                        node.out_shapes.append(value_info['shape'])
                else:
                    if self.is_inference == False:
                        self.get_results_of_inference(
                            onnx_model, value_infos,
                            self.decoder.onnx_graph.place_holder_nodes)
                        self.is_inference = True
                    _, dtype, shape = self.get_dynamic_shape(opt)
                    node.dtype = dtype
                    node.out_shapes.append(shape)

    def remove_tmp_data(self):
        """
        remove temporarily generated file
        """
        if os.path.exists(self.tmp_data_dir):
            import shutil
            shutil.rmtree(self.tmp_data_dir)

    def directly_map(self, node, name='', *args, **kwargs):
        inputs = node.layer.input
        outputs = node.layer.output
        op_type = node.layer_type
        attrs = node.attr_map
        info = default_op_mapping[op_type]
        info.extend(list(default_op_mapping_field_values.values())[len(info):])
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

        if fluid_op in default_ioa_constraint:
            for predicate, message in default_ioa_constraint[fluid_op]:
                assert predicate(inputs, outputs, attrs), message

        mapped_attrs = {
            attr_mapping.get(key, key): value
            for key, value in attrs.items()
        }
        if '' in mapped_attrs:
            mapped_attrs.pop('')
        if '_' in mapped_attrs:
            mapped_attrs.pop('_')
        fluid_attrs = default_attrs.copy()
        fluid_attrs.update(mapped_attrs)
        inputs = inputs if input_perm is None else list(
            map(lambda i: inputs[i], input_perm))
        val_inps = []
        for idx, ipt in enumerate(inputs):
            val_inps.append(self.graph.get_input_node(node, idx=idx, copy=True))

        val_outs = outputs if output_perm is None else list(
            map(lambda i: outputs[i], output_perm))
        attr = fluid_attrs
        assert len(val_inps) == 1, 'directly_map error with multi inputs'
        if fluid_op not in ['shape']:
            attr['name'] = string(node.layer_name)
        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_inps[0],
                                  output=val_outs[0],
                                  param_attr=attr)

    def deal_custom_layer(self, node):
        op = node.layer_type
        custom_code, func = make_custom_layer(node)
        child_func_code, child_func = make_custom_child_func(node)
        params = get_params(node.layer, node.layer_type)
        arg_names, kwargs = set_args(func, params)
        kwargs['name'] = string(node.layer_name)
        node.fluid_code.add_layer(func.__code__.co_name,
                                  inputs=node.inputs,
                                  output=node,
                                  param_attr=kwargs,
                                  is_custom_layer=True)
        if op not in self.used_custom_layers:
            self.used_custom_layers[op] = custom_code
            if op + '_child_func' not in self.used_custom_layers:
                if child_func_code is not None:
                    self.used_custom_layers[op +
                                            '_child_func'] = child_func_code

    def elementwise_map(self, node):
        assert node.layer_type in self.elementwise_ops
        op_type = self.elementwise_ops[node.layer_type]

        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        val_y_shape = val_y.out_shapes[0]
        val_x_shape = val_x.out_shapes[0]

        if len(val_x_shape) < len(val_y_shape):
            val_x, val_y = val_y, val_x

        str_y_shape = ','.join(str(e) for e in val_y_shape)
        str_x_shape = ','.join(str(e) for e in val_x_shape)
        slice_idx = 0
        if str_y_shape not in str_x_shape:
            for dim in val_y_shape:
                if dim == 1:
                    slice_idx += 1
                else:
                    break
        attr = {"name": string(node.layer_name)}
        if slice_idx < len(val_y_shape) and slice_idx > 0:
            val_y_reshaped = val_y_shape[slice_idx:]
            var_y_reshaped = val_y.layer_name + '_reshaped'
            attr_reshaped = {
                'shape': val_y_reshaped,
                'name': string(var_y_reshaped)
            }
            node.fluid_code.add_layer('reshape',
                                      inputs=val_y,
                                      output=var_y_reshaped,
                                      param_attr=attr_reshaped)
            inputs = {'x': val_x, 'y': var_y_reshaped}
            node.fluid_code.add_layer(op_type,
                                      inputs=inputs,
                                      output=node,
                                      param_attr=attr)
        else:
            inputs = {'x': val_x, 'y': val_y}
            node.fluid_code.add_layer(op_type,
                                      inputs=inputs,
                                      output=node,
                                      param_attr=attr)

    def place_holder(self, node):
        self.input_shapes.append(node.out_shapes[0])

        shape = node.out_shapes[0]
        for i, dim_shape in enumerate(shape):
            if dim_shape == 0 and i == 0:
                shape[i] = 1
            if dim_shape == 0 and i != 0:
                assert 'shape of input is not assigned'
        attr = {
            "dtype": string(node.dtype),
            "shape": shape,
            "name": string(node.layer_name),
            "append_batch_size": 'False'
        }

        node.fluid_code.add_layer("data",
                                  inputs=None,
                                  output=node,
                                  param_attr=attr)

    def create_parameter(self, node, parameter=None):
        if parameter is not None:
            node = parameter
        dtype = node.dtype
        shape = node.out_shapes[0]
        if len(node.weight.shape) == 0:
            shape = [1]
        self.weights[node.layer_name] = node.weight
        attr = {
            'dtype': string(dtype),
            'shape': shape,
            'name': string(node.layer_name),
            'attr': string(node.layer_name),
            'default_initializer': 'Constant(0.0)'
        }
        node.fluid_code.add_layer("create_parameter",
                                  inputs=None,
                                  output=node,
                                  param_attr=attr)

    def _pad_if_asymmetric(self, node, pads, val_name):  # pads: SSEE
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

    def _interpolate(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_scales = self.graph.get_input_node(node, idx=1, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)
        out_shape = val_y.out_shapes[0]
        if out_shape is not None:
            assert len(out_shape) == 4, 'only 4-D Tensor as X and Y supported'
            out_shape = out_shape[2:]

        scales = _const_weight_or_none(val_scales)

        if isinstance(val_scales, ONNXGraphNode):
            scales, _, _ = self.get_dynamic_shape(val_scales.layer_name)
        attr = {'name': string(node.layer_name)}
        use_scales = True
        if scales is not None:
            try:
                assert len(scales) == 4, 'only 4-D Tensor as X and Y supported'
                assert scales[0] == 1 and scales[
                    1] == 1, 'only scale on (NC)HW supported'
                assert scales[2] == scales[
                    3], 'only aspect-ratio-invariant scale supported'
            except:
                use_scales = False
        scale = scales[2] if scales else None
        if scale is None:
            assert out_shape, 'neither scales nor output shape is available'
        else:
            if out_shape is None:
                in_shape = val_x.out_shapes[0]
                assert in_shape is not None, 'out_shape required but not inferrable'
                assert len(
                    in_shape) == 4, 'only 4-D Tensor as X and Y supported'
                out_shape = [in_shape[2] * scale, in_shape[3] * scale]

        mode = node.get_attr('mode', 'nearest')

        fluid_op = 'resize_{}'.format(mode)
        if 'linear' in mode:
            print(
                'Warnning: paddle not support op:resize wiht mode: linear, we use bilinear replace linear'
            )
            fluid_op = 'resize_bilinear'

        if use_scales and scale is not None:
            attr['scale'] = scale
        else:
            attr['out_shape'] = out_shape

        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def RoiAlign(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_rois = self.graph.get_input_node(node, idx=1, copy=True)

        pooled_height = node.get_attr('output_height')
        pooled_width = node.get_attr('output_width')
        spatial_scale = node.get_attr('spatial_scale')
        sampling_ratio = node.get_attr('sampling_ratio')
        attr = {
            'pooled_height': pooled_height,
            'pooled_width': pooled_width,
            'spatial_scale': spatial_scale,
            'sampling_ratio': sampling_ratio,
        }
        node.fluid_code.add_layer('roi_align',
                                  inputs={
                                      'input': val_x,
                                      'rois': val_rois
                                  },
                                  output=node,
                                  param_attr=attr)

    def MaxRoiPool(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_rois = self.graph.get_input_node(node, idx=1, copy=True)

        spatial_scale = node.get_attr('spatial_scale')
        pooled_height, pooled_width = node.get_attr('pooled_shape')
        attr = {
            'pooled_height': pooled_height,
            'pooled_width': pooled_width,
            'spatial_scale': spatial_scale,
        }
        node.fluid_code.add_layer('roi_pool',
                                  inputs={
                                      'input': val_x,
                                      'rois': val_rois
                                  },
                                  output=node,
                                  param_attr=attr)

    def Pad(self, node, op_independent=True):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        pads = node.get_attr('pads')
        mode = node.get_attr('mode', 'constant')
        value = node.get_attr('value', 0.)
        data_shape = val_x.out_shapes[0]
        output_shape = node.out_shapes[0]
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
            fluid_op = 'pad'
        if len(pads) == 4:
            paddings = np.array(pads).reshape(
                (-1, 2)).transpose().flatten().tolist()  # SSEE -> SESE
        elif len(pads) == 8:
            paddings = np.array(pads).reshape(
                (-1, 4)).transpose().flatten().tolist()  # SSEE -> SESE
            if sum(paddings[:4]) == 0:
                fluid_op = 'pad2d'
                paddings = paddings[4:]
                attr['mode'] = string(mode)
        attr['paddings'] = paddings
        if op_independent:
            attr['name'] = string(node.layer_name)
            node.fluid_code.add_layer(fluid_op,
                                      inputs=val_x,
                                      output=node,
                                      param_attr=attr)
        else:
            attr['name'] = string(node.layer_name + '_paded')
            node.fluid_code.add_layer(fluid_op,
                                      inputs=val_x,
                                      output=node.layer_name + '_paded',
                                      param_attr=attr)
            return node.layer_name + '_paded'

    def Unsqueeze(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = node.get_attr('axes')
        if len(val_x.out_shapes[0]) == 0:
            node.fluid_code.add_layer('assign',
                                      inputs=val_x,
                                      output=node,
                                      param_attr=None)
        else:
            attr = {'axes': axes, 'name': string(node.layer_name)}
            node.fluid_code.add_layer('unsqueeze',
                                      inputs=val_x,
                                      output=node,
                                      param_attr=attr)

    def Shrink(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        bias = node.get_attr('bias')
        lambd = node.get_attr('lambd')
        assert bias == 0.0, 'not support bias!=0'
        attr = {'threshold': lambd, 'name': node.layer_name}
        node.fluid_code.add_layer('hard_shrink',
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Constant(self, node):
        val_output = self.graph.get_node(node.layer.output[0], copy=True)

        value = node.get_attr('value')
        dtype = np.dtype(value.dtype)
        output_dtype = val_output.dtype
        if output_dtype:
            assert dtype == output_dtype, 'tensor dtype unmatches storage dtype'

        shape = node.get_attr('shape', None)

        if shape is None:
            shape = val_output.out_shapes[0]
        if shape is None:
            shape = list(value.shape)
            _logger.warning(
                'in (Constant -> %s): '
                'attribute "shape" of %s not inferred, '
                'using value as 1-D tensor may lead to fails',
                val_output.layer_name, val_output.layer_name)

        if len(value) == 1:
            value = value.tolist()
            shape = [1]
            value = value[0]
            if dtype.name == 'int64':
                dtype = 'int32'
            attr = {'shape': shape, 'dtype': string(dtype), 'value': value}
            node.fluid_code.add_layer('fill_constant',
                                      inputs=None,
                                      output=node,
                                      param_attr=attr)
        else:
            value = np.reshape(value, shape)
            self.weights[node.layer_name] = value
            attr = {
                'dtype': string(dtype),
                'shape': shape,
                'name': string(node.layer_name),
                'attr': string(node.layer_name),
                'default_initializer': 'Constant(0.0)'
            }
            node.fluid_code.add_layer("create_parameter",
                                      inputs=None,
                                      output=node,
                                      param_attr=attr)

    def Resize(self, node):
        self._interpolate(node)

    def Upsample(self, node):
        self._interpolate(node)

    def Expand(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_shape = self.graph.get_input_node(node, idx=1, copy=True)

        if len(val_shape.outputs) == 1:
            self.omit_nodes.append(val_shape.layer_name)

        val_y = self.graph.get_node(node.layer.output[0], copy=True)
        out_shape = node.out_shapes[0]
        val_x_dtype = val_x.dtype

        name_ones = node.layer_name + '_ones'
        attr_ones = {'shape': out_shape, 'dtype': string(val_x_dtype)}
        node.fluid_code.add_layer('ones',
                                  inputs=None,
                                  output=name_ones,
                                  param_attr=attr_ones)
        inputs = {'x': name_ones, 'y': val_x}
        attr = {'name': string(node.layer_name)}
        node.fluid_code.add_layer('elementwise_mul',
                                  inputs=inputs,
                                  output=node.layer_name,
                                  param_attr=attr)

    def Gather(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        indices = self.graph.get_input_node(node, idx=1, copy=True)
        indices_shape = indices.out_shapes[0]
        axis = node.get_attr('axis', 0)
        assert len(
            indices_shape) <= 2, "Gather op don't support dim of indice >2 "
        if axis == 0 and len(indices_shape) <= 1:
            node.fluid_code.add_layer('gather',
                                      inputs={
                                          'input': val_x,
                                          'index': indices
                                      },
                                      output=node,
                                      param_attr=None)
        elif axis > 0 and len(indices_shape) <= 1:
            perm = list(range(len(val_x.out_shapes[0])))
            perm = [axis] + perm[:axis] + perm[axis + 1:]
            attr_trans = {'perm': perm}
            name_trans = val_x.layer_name + '_trans'
            node.fluid_code.add_layer('transpose',
                                      inputs=val_x,
                                      output=name_trans,
                                      param_attr=attr_trans)
            node.fluid_code.add_layer('gather',
                                      inputs={
                                          'input': name_trans,
                                          'index': indices
                                      },
                                      output=node,
                                      param_attr=None)
            node.fluid_code.add_layer('transpose',
                                      inputs=node,
                                      output=node,
                                      param_attr=attr_trans)
        elif len(indices_shape) > 1:
            from functools import reduce
            reshape_shape = reduce(lambda x, y: x * y, indices_shape)
            node.fluid_code.add_layer('reshape',
                                      inputs=indices,
                                      output=indices,
                                      param_attr={'shape': [
                                          reshape_shape,
                                      ]})

            perm = list(range(len(val_x.out_shapes[0])))
            perm = [axis] + perm[:axis] + perm[axis + 1:]
            attr_trans = {'perm': perm}
            name_trans = val_x.layer_name + '_trans'
            node.fluid_code.add_layer('transpose',
                                      inputs=val_x,
                                      output=name_trans,
                                      param_attr=attr_trans)
            node.fluid_code.add_layer('gather',
                                      inputs={
                                          'input': name_trans,
                                          'index': indices
                                      },
                                      output=node,
                                      param_attr=None)
            node.fluid_code.add_layer('transpose',
                                      inputs=node,
                                      output=node,
                                      param_attr=attr_trans)
            val_x_shape = val_x.out_shapes[0]
            reshaped_shape = []
            for i in perm:
                reshaped_shape.append(indices_shape[i])
            for i in val_x_shape[:axis] + val_x_shape[axis + 1:]:
                reshaped_shape.append(i)
            node.fluid_code.add_layer('reshape',
                                      inputs=node,
                                      output=node,
                                      param_attr={'shape': reshaped_shape})

    def Slice(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        starts, ends, axes, steps = None, None, None, None
        if len(node.inputs) > 1:
            starts = self.graph.get_input_node(node, idx=1, copy=True)
            ends = self.graph.get_input_node(node, idx=2, copy=True)
            if len(node.inputs) > 3:
                axes = self.graph.get_input_node(node, idx=3, copy=True)
                self.omit_nodes.append(axes.layer_name)
                axes = _const_weight_or_none(axes)
            if len(node.inputs) > 4:
                steps = self.graph.get_input_node(node, idx=4, copy=True)
                self.omit_nodes.append(steps.layer_name)
                steps = _const_weight_or_none(steps)

            self.omit_nodes.append(starts.layer_name)
            self.omit_nodes.append(ends.layer_name)
            starts = _const_weight_or_none(starts).copy()
            ends = _const_weight_or_none(ends).copy()
        else:
            starts = node.get_attr('starts')
            ends = node.get_attr('ends')
            axes = node.get_attr('axes')

        val_y = self.graph.get_node(node.layer.output[0], copy=True)

        shape = val_x.out_shapes[0]

        if shape is not None:
            for idx, value in enumerate(starts):
                if value > shape[axes[idx]]:
                    starts[idx] = shape[axes[idx]]
            for idx, value in enumerate(ends):
                if value > shape[axes[idx]]:
                    ends[idx] = shape[axes[idx]]
        attr = {"axes": axes, "starts": starts, "ends": ends}
        node.fluid_code.add_layer('slice',
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def ConstantOfShape(self, node):
        val_shape = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)
        shape = _const_weight_or_none(val_shape)

        if shape is None:
            shape = node.out_shapes[0]

        assert shape is not None, (
            'given shape is neither const value nor deductible from output, '
            'this is not supported')

        value = node.get_attr('value')
        dtype = value.dtype
        value = value.tolist()
        if len(value) == 1:
            shape = [1]
            value = value[0]
            if dtype.name == 'int64':
                dtype = 'int32'
            attr = {'shape': shape, 'dtype': string(dtype), 'value': value}
            node.fluid_code.add_layer('fill_constant',
                                      inputs=None,
                                      output=node,
                                      param_attr=attr)

    def Split(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)

        fluid_op = 'split'
        split = node.get_attr('split')
        axis = node.get_attr('axis', 0)
        attr = {
            'num_or_sections': split,
            'dim': axis,
            'name': string(node.layer_name)
        }

        node.fluid_code.add_layer('split',
                                  inputs=val_x,
                                  output=val_y,
                                  param_attr=attr)

    def Reshape(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_shape = self.graph.get_input_node(node, idx=1, copy=True)
        val_reshaped = self.graph.get_node(node.layer.output[0], copy=True)
        shape = None

        if isinstance(val_shape, ONNXGraphDataNode):
            self.omit_nodes.append(val_shape.layer_name)

        attr = {'name': string(node.layer_name)}
        # catch dynamic graph shape
        if isinstance(val_shape, ONNXGraphNode):
            shape, _, _ = self.get_dynamic_shape(val_shape.layer_name)
            if val_shape.dtype == 'int64':
                val_shape_cast = val_shape.layer_name + '_cast'
                node.fluid_code.add_layer('cast',
                                          inputs=val_shape,
                                          output=val_shape_cast,
                                          param_attr={'dtype': string('int32')})

                attr['actual_shape'] = val_shape_cast
            else:
                attr['actual_shape'] = val_shape

        if shape is None:
            shape = val_reshaped.out_shapes[0]

        if shape is None:
            shape = [1, -1]
            _logger.warning(
                'in %s(%s -> Reshape -> %s): '
                'input "shape" not inferred, use [1, -1] as dummy value, '
                'the behavior of Paddle fluid maybe undefined', node.layer_name,
                val_x.layer_name, val_reshaped.layer_name)

        attr['shape'] = shape
        node.fluid_code.add_layer('reshape',
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Cast(self, node):
        val_input = self.graph.get_input_node(node, idx=0, copy=True)
        val_output = self.graph.get_node(node.layer.output[0], copy=True)

        dtype = node.get_attr('to')
        if not isinstance(dtype, np.dtype):
            dtype = TENSOR_TYPE_TO_NP_TYPE[dtype]

        output_dtype = val_output.dtype
        if output_dtype:
            assert dtype == output_dtype, 'dtype of to unmatches output'
        attr = {'dtype': string(dtype)}
        node.fluid_code.add_layer('cast',
                                  inputs=val_input,
                                  output=node,
                                  param_attr=attr)

    def AveragePool(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)

        auto_pad = node.get_attr('auto_pad', 'NOTSET')
        kernel_shape = node.get_attr("kernel_shape")
        poolnd = len(kernel_shape)
        strides = node.get_attr("strides")
        pad_mode = node.get_attr("pads")
        ceil_mode = bool(node.get_attr('ceil_mode', 0))
        pads = node.get_attr('pads', [0] * (poolnd * 2))
        fluid_op = 'pool{}d'.format(poolnd)
        assert 2 <= poolnd <= 3, 'only pool2d and pool3d is supported'

        paddings, val_x = self._pad_if_asymmetric(node, pads, val_x)

        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
            input_shape = val_x.out_shapes[0]
            pad_h = get_same_padding(input_shape[2], kernel_shape[0],
                                     strides[0])
            pad_w = get_same_padding(input_shape[3], kernel_shape[1],
                                     strides[1])
            attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}

        attr = {
            "pool_size": kernel_shape,
            "pool_type": string('avg'),
            "pool_stride": strides,
            "pool_padding": paddings,
            "ceil_mode": ceil_mode,
            "exclusive": 'True',
            "name": string(node.layer_name)
        }

        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Concat(self, node):
        inputs = []
        for i in range(len(node.layer.input)):
            ipt = self.graph.get_input_node(node, idx=i, copy=True)
            if isinstance(ipt, str):
                inputs.append(ipt)
            else:
                inputs.append(ipt.layer_name)
        axis = node.get_attr('axis')
        attr = {'axis': axis}
        node.fluid_code.add_layer('concat',
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)

    def Flatten(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axis = node.get_attr('axis', 1)
        attr = {"axis": str(axis), "name": string(node.layer_name)}
        node.fluid_code.add_layer('flatten',
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Gemm(self, node):
        val_a = self.graph.get_input_node(node, idx=0, copy=True)
        val_b = self.graph.get_input_node(node, idx=1, copy=True)
        val_c = self.graph.get_input_node(node, idx=2, copy=True)

        alpha = node.get_attr('alpha', 1.)  # optional
        beta = node.get_attr('beta', 1.)  # optional
        trans_a = bool(node.get_attr('transA', 0))  # optional
        trans_b = bool(node.get_attr('transB', 0))  # optional
        val_mm = node.layer_name + '_mm'
        matmul_inputs = {"x": val_a, "y": val_b}
        attr_matmul = {
            "transpose_x": trans_a,
            "transpose_y": trans_b,
            "alpha": alpha,
            "name": string(val_mm)
        }
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
                var_beta = node.layer_name + '_beta'
                matmul_beta_inputs = {"x": val_c, "y": var_beta}
                node.fluid_code.add_layer("Constant",
                                          inputs=matmul_beta_inputs,
                                          output=var_beta,
                                          param_attr={'value': beta})

                add_inputs = {"x": val_mm, "y": var_beta}
                attr = {"name": string(node.layer_name)}
                node.fluid_code.add_layer("elementwise_add",
                                          inputs=add_inputs,
                                          output=node,
                                          param_attr=attr)

    def Sum(self, node):
        val_inps = node.layer.input
        inputs = {
            "x": self.graph.get_input_node(node, idx=0, copy=True),
            "y": self.graph.get_input_node(node, idx=1, copy=True),
        }
        node.fluid_code.add_layer("elementwise_add", inputs=inputs, output=node)

        for idx, ipt in enumerate(val_inps[2:]):
            y = self.graph.get_input_node(node, idx=idx, copy=True)
            inputs = {
                "x": node.layer_name,
                "y": y,
            }
            node.fluid_code.add_layer("elementwise_add",
                                      inputs=inputs,
                                      output=node)

    def MatMul(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        inputs = {"x": val_x, "y": val_y}
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer("matmul",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)

    def BatchNormalization(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_scale = self.graph.get_input_node(node, idx=1, copy=True)
        val_b = self.graph.get_input_node(node, idx=2, copy=True)
        val_mean = self.graph.get_input_node(node, idx=3, copy=True)
        val_var = self.graph.get_input_node(node, idx=4, copy=True)

        self.omit_nodes.append(val_scale.layer_name)
        self.omit_nodes.append(val_b.layer_name)
        self.omit_nodes.append(val_mean.layer_name)
        self.omit_nodes.append(val_var.layer_name)

        momentum = node.get_attr('momentum', .9)
        epsilon = node.get_attr('epsilon', 1e-5)

        # Attribute: spatial is used in BatchNormalization-1,6,7
        spatial = bool(node.get_attr('spatial'))
        attr = {
            "momentum": momentum,
            "epsilon": epsilon,
            "data_layout": string('NCHW'),
            "is_test": True,
            "param_attr": string(val_scale.layer_name),
            "bias_attr": string(val_b.layer_name),
            "moving_mean_name": string(val_mean.layer_name),
            "moving_variance_name": string(val_var.layer_name),
            "use_global_stats": spatial,
            "name": string(node.layer_name)
        }
        node.fluid_code.add_layer("batch_norm",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Transpose(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        perm = node.get_attr('perm')
        attr = {'perm': perm, "name": string(node.layer_name)}
        node.fluid_code.add_layer("transpose",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Relu(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer("relu",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def PRelu(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_slope = self.graph.get_input_node(node, idx=1, copy=True)

        mode = 'channel'
        shape_slope = val_slope.out_shapes[0]
        if len(shape_slope) == 1:
            mode = 'all'
        elif len(shape_slope) > 2:
            mode = 'element'
        attr = {
            "param_attr": string(val_slope.layer_name),
            'mode': string(mode)
        }
        node.fluid_code.add_layer("prelu",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Squeeze(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        axes = node.get_attr('axes')
        attr = {'axes': axes, "name": string(node.layer_name)}
        node.fluid_code.add_layer("squeeze",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Equal(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_input_node(node, idx=1, copy=True)
        node.fluid_code.add_layer("equal",
                                  inputs={
                                      'x': val_x,
                                      'y': val_y
                                  },
                                  output=node,
                                  param_attr=None)

    def Where(self, node):
        condition = self.graph.get_input_node(node, idx=0, copy=True)
        val_x = self.graph.get_input_node(node, idx=1, copy=True)
        val_y = self.graph.get_input_node(node, idx=2, copy=True)

        not_condition = condition.layer_name + '_not'
        node.fluid_code.add_layer("logical_not",
                                  inputs=condition,
                                  output=not_condition,
                                  param_attr=None)
        cast_not_condition = not_condition + '_cast'
        node.fluid_code.add_layer("cast",
                                  inputs=not_condition,
                                  output=cast_not_condition,
                                  param_attr={'dtype': string(val_x.dtype)})
        cast_condition = condition.layer_name + '_cast'
        node.fluid_code.add_layer("cast",
                                  inputs=condition,
                                  output=cast_condition,
                                  param_attr={'dtype': string(val_x.dtype)})
        mul_val_x = val_x.layer_name + '_mul'
        node.fluid_code.add_layer("elementwise_mul",
                                  inputs={
                                      'x': val_x,
                                      'y': cast_condition
                                  },
                                  output=mul_val_x,
                                  param_attr=None)

        mul_val_y = val_y.layer_name + '_mul'
        node.fluid_code.add_layer("elementwise_mul",
                                  inputs={
                                      'x': val_y,
                                      'y': cast_not_condition
                                  },
                                  output=mul_val_y,
                                  param_attr=None)

        node.fluid_code.add_layer("elementwise_add",
                                  inputs={
                                      'x': mul_val_x,
                                      'y': mul_val_y
                                  },
                                  output=node,
                                  param_attr=None)

    def NonZero(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        where_name = node.layer_name + '_where'
        node.fluid_code.add_layer("where",
                                  inputs=val_x.layer_name + '!=0',
                                  output=where_name)
        dims = len(val_x.out_shapes[0])
        elements_count_val_x = reduce(lambda x, y: x * y, val_x.out_shapes[0])
        flatten_names = []
        for dim in range(dims):
            slice_name = node.layer_name + '_slice' + str(dim)
            flatten_name = node.layer_name + '_flatten' + str(dim)
            flatten_names.append(flatten_name)
            attr = {
                'axes': list(range(dims)),
                'starts': [0, dim],
                'ends': [elements_count_val_x, dim + 1]
            }
            node.fluid_code.add_layer("slice",
                                      inputs=where_name,
                                      output=slice_name,
                                      param_attr=attr)
            node.fluid_code.add_layer("flatten",
                                      inputs=slice_name,
                                      output=flatten_name,
                                      param_attr={'axis': 0})
        node.fluid_code.add_layer("concat",
                                  inputs=flatten_names,
                                  output=node,
                                  param_attr={'axis': 0})

    def Identity(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        node.fluid_code.add_layer("assign", inputs=val_x, output=node)

    def Tile(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_repeats = self.graph.get_input_node(node, idx=1, copy=True)
        repeats = _const_weight_or_none(val_repeats)
        assert repeats is not None, 'for OP:Tile, only const repeats supported'

        if isinstance(repeats, int):
            repeats = [repeats]

        attr = {
            'expand_times': repeats,
            "name": string(node.layer_name),
        }
        node.fluid_code.add_layer("expand",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def MaxPool(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)

        auto_pad = node.get_attr('auto_pad', 'NOTSET')
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

        paddings, val_x = self._pad_if_asymmetric(node, pads, val_x)

        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
            input_shape = val_x.out_shapes[0]
            pad_h = get_same_padding(input_shape[2], kernel_shape[0],
                                     strides[0])
            pad_w = get_same_padding(input_shape[3], kernel_shape[1],
                                     strides[1])
            attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}

        attr = {
            "pool_size": kernel_shape,
            "pool_type": string("max"),
            "pool_stride": strides,
            "pool_padding": paddings,
            "ceil_mode": ceil_mode,
            "name": string(node.layer_name),
            "exclusive": False
        }
        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def _global_pool(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)
        input_shape = val_x.out_shapes[0]
        output_shape = val_y.out_shapes[0]
        assert input_shape is not None or output_shape is not None, 'poolnd not inferred'  # N
        if input_shape:
            poolnd = len(input_shape) - 2  # NC...
        elif output_shape:
            poolnd = len(output_shape) - 2  # NC...
        assert 2 <= poolnd <= 3, 'only pool2d and pool3d is supported'
        fluid_op = 'pool{}d'.format(poolnd)

        pool_type = None
        if node.layer.op_type == 'GlobalMaxPool':
            pool_type = 'max'
        elif node.layer.op_type == 'GlobalAveragePool':
            pool_type = 'avg'

        attr = {
            "pool_type": string(pool_type),
            "global_pooling": True,
            "name": string(node.layer_name)
        }
        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def GlobalMaxPool(self, node):
        self._global_pool(node)

    def GlobalAveragePool(self, node):
        self._global_pool(node)

    def Conv(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_w = self.graph.get_input_node(node, idx=1, copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)

        self.omit_nodes.append(val_w.layer_name)

        has_bias = len(node.layer.input) == 3
        if has_bias:
            val_b = self.graph.get_input_node(node, idx=2, copy=True)
            self.omit_nodes.append(val_b.layer_name)
        auto_pad = node.get_attr('auto_pad', 'NOTSET')

        kernel_shape = node.get_attr('kernel_shape')
        convnd = len(kernel_shape)
        assert 2 <= convnd <= 3, 'only conv2d and conv3d is supported'
        num_out_channels = val_w.out_shapes[0][0]  # OI...
        fluid_op = 'conv{}d'.format(convnd)

        num_groups = node.get_attr('group', 1)
        strides = node.get_attr('strides', [1] * convnd)  # optional
        dilations = node.get_attr('dilations', [1] * convnd)  # optional
        pads = node.get_attr('pads', [0] * (convnd * 2))  # optional

        input_shape = val_x.out_shapes[0]
        paddings, val_x = self._pad_if_asymmetric(node, pads, val_x)

        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
            pad_h = get_same_padding(input_shape[2], kernel_shape[0],
                                     strides[0])
            pad_w = get_same_padding(input_shape[3], kernel_shape[1],
                                     strides[1])
            attr = {"paddings": pad_h + pad_w, "pad_value": 0.0}

        attr = {
            "num_filters": num_out_channels,
            "filter_size": kernel_shape,
            "stride": strides,
            "padding": paddings,
            "dilation": dilations,
            "groups": num_groups,
            'param_attr': string(val_w.layer_name),
            "name": string(node.layer_name)
        }
        if has_bias:
            attr["bias_attr"] = string(val_b.layer_name)
        else:
            attr["bias_attr"] = False
        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def ConvTranspose(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_w = self.graph.get_input_node(node, idx=1, copy=True)
        val_b = None
        if len(node.layer.input) > 2:
            val_b = self.graph.get_input_node(node, idx=2, copy=True)
            self.omit_nodes.append(val_b.layer_name)
        self.omit_nodes.append(val_w.layer_name)

        val_y = self.graph.get_node(node.layer.output[0], copy=True)

        auto_pad = node.get_attr('auto_pad', 'NOTSET')
        out_padding = node.get_attr('output_padding', [0, 0])
        kernel_shape = node.get_attr('kernel_shape')
        assert kernel_shape, 'kernel_shape not inferred'
        convnd = len(kernel_shape)
        assert 2 <= convnd <= 3, 'only conv2d_transpose and conv3d_transpose supported'
        num_out_channels = val_w.out_shapes[0][1]
        fluid_op = 'conv{}d_transpose'.format(convnd)

        num_groups = node.get_attr('group', 1)
        strides = node.get_attr('strides', [1] * convnd)
        dilations = node.get_attr('dilations', [1] * convnd)
        output_size = node.get_attr('output_shape', [])
        pads = node.get_attr('pads', [0] * (convnd * 2))

        paddings, var_x = self._pad_if_asymmetric(node, pads, val_x)

        output_size = [0, 0]

        output_size[0] = (val_x.out_shapes[0][2] -
                          1) * strides[0] - 2 * paddings[0] + dilations[0] * (
                              kernel_shape[0] - 1) + 1 + out_padding[0]
        output_size[1] = (val_x.out_shapes[0][3] -
                          1) * strides[1] - 2 * paddings[1] + dilations[1] * (
                              kernel_shape[1] - 1) + 1 + out_padding[1]
        attr = {
            'num_filters': num_out_channels,
            'output_size': output_size or None,
            'filter_size': kernel_shape,
            'padding': paddings,
            'stride': strides,
            'dilation': dilations,
            'groups': num_groups,
            'param_attr': string(val_w.layer_name),
            'bias_attr': None if val_b is None else string(val_b.layer_name),
            'name': string(node.layer_name),
        }
        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def GRU(self, node):
        val_x = self.graph.get_input_node(node, idx=0, copy=True)
        val_w = self.graph.get_input_node(node, idx=1, copy=True)
        val_r = self.graph.get_input_node(node, idx=2, copy=True)

        val_b = None
        val_len = None
        val_xh = None
        miss_arg_num = 0
        num_ipt = len(node.layer.input)
        if num_ipt > 3 and node.layer.input[3] != '':
            val_b = self.graph.get_input_node(node, idx=3, copy=True)
        else:
            miss_arg_num += 1
        if num_ipt > 4 and node.layer.input[4] != '':
            val_len = self.graph.get_input_node(node,
                                                idx=4 - miss_arg_num,
                                                copy=True)
        else:
            miss_arg_num += 1
        if num_ipt > 5 and node.layer.input[5] != '':
            val_xh = self.graph.get_input_node(node,
                                               idx=5 - miss_arg_num,
                                               copy=True)

        data, dtype, shape = self.get_dynamic_shape(val_x.layer_name)

        x_shape = val_x.out_shapes[0]

        assert x_shape[1] == 1, 'only X with batch_size = 1 supported'
        assert node.get_attr('clip', None) is None, 'clipping not supported'

        hidden_size = node.get_attr('hidden_size', None)
        if hidden_size is None:
            r_shape = val_r.out_shapes[0]
            if r_shape:
                hidden_size = r_shape[-1]
        if hidden_size is None:
            w_shape = var_w.out_shapes[0]
            if w_shape:
                hidden_size = w_shape[-2] // 3
        if hidden_size is None and val_b:
            b_shape = val_b.out_shapes[0]
            if b_shape:
                hidden_size = b_shape[-1] // 6
        if hidden_size is None and val_xh:
            xh_shape = val_xh.out_shapes[0]
            if xh_shape:
                hidden_size = xh_shape[-1]

        direction = node.get_attr('direction', 'forward')
        assert direction != 'bidirectional', 'direction = bidirectional not supported'

        activations = node.get_attr('activations', ['Sigmoid', 'Tanh'])
        assert len(activations) == 2, 'bidirectional operation not supported'

        assert node.get_attr('linear_before_reset',
                             0) == 0, 'only linear_before_reset = 0 supported'

        activations = [s.lower() for s in activations]
        gate_activation, candidate_activation = activations
        is_reverse = direction == 'reverse'

        var_x0 = node.layer_name + '_x0'
        node.fluid_code.add_layer('squeeze',
                                  inputs=val_x,
                                  output=var_x0,
                                  param_attr={
                                      'axes': [1],
                                      'name': string(var_x0)
                                  })

        var_w0 = node.layer_name + '_w0'
        node.fluid_code.add_layer('squeeze',
                                  inputs=val_w,
                                  output=var_w0,
                                  param_attr={
                                      'axes': [0],
                                      'name': string(var_w0)
                                  })

        var_fc = node.layer_name + '_fc'
        var_mm = (node.layer_name + '_mm') if val_b else var_fc
        node.fluid_code.add_layer('matmul',
                                  inputs={
                                      'x': var_x0,
                                      'y': var_w0
                                  },
                                  output=var_mm,
                                  param_attr={
                                      'transpose_x': 0,
                                      'transpose_y': 1,
                                      'name': string(var_mm)
                                  })

        var_r0 = node.layer_name + '_r0'
        node.fluid_code.add_layer('squeeze',
                                  inputs=val_r,
                                  output=var_r0,
                                  param_attr={
                                      'axes': [0],
                                      'name': string(var_r0)
                                  })

        var_r0t = node.layer_name + '_r0t'

        node.fluid_code.add_layer('transpose',
                                  inputs=var_r0,
                                  output=var_r0t,
                                  param_attr={
                                      'perm': [1, 0],
                                      'name': string(var_r0t)
                                  })
        if val_b:
            var_bi = node.layer_name + '_bi'
            var_bh = node.layer_name + '_bh'
            node.fluid_code.add_layer('split',
                                      inputs=val_b,
                                      output=var_bi + ',' + var_bh,
                                      param_attr={
                                          'axis':
                                          1,
                                          'split':
                                          [hidden_size * 3, hidden_size * 3],
                                          'name':
                                          string(node.layer_name + '.b/split')
                                      })
            var_bi0 = node.layer_name + '_bi0'
            node.fluid_code.add_layer('squeeze',
                                      inputs=var_bi,
                                      output=var_bi0,
                                      param_attr={
                                          'axes': [0],
                                          'name': string(var_bi0)
                                      })

            node.fluid_code.add_layer('elmentwise_add',
                                      inputs=[var_mm, var_bi0],
                                      output=var_fc,
                                      param_attr={
                                          'axes':
                                          1,
                                          'name':
                                          string(node.layer_name + '.i/bias')
                                      })

        if val_xh:
            var_xh0 = node.layer_name + '_xh0'
            node.fluid_code.add_layer('squeeze',
                                      inputs=val_xh,
                                      output=var_xh0,
                                      param_attr={
                                          'axes': [1],
                                          'name': string(var_xh0)
                                      })
        var_y00 = node.layer_name + '_y00'

        attr = {
            'origin_mode': True,
            'h_0': var_xh0 if val_xh else None,
            'is_reverse': is_reverse,
            'gate_activation': string(gate_activation),
            'candidate_activation': string(candidate_activation),
            'param_attr': string(var_r0t),
            'bias_attr': string(var_bh) if val_b else False,
        }
        node.fluid_code.add_layer('dynamic_gru',
                                  inputs=var_fc + ',' + str(hidden_size),
                                  output=var_y00,
                                  param_attr=attr)

        num_opt = len(node.layer.output)

        if num_opt > 0 and node.layer.output[0] != '':
            node.fluid_code.add_layer('unsqueeze',
                                      inputs=var_y00,
                                      output=node.layer.output[0],
                                      param_attr={
                                          'axes': [1, 1],
                                          'name': string(node.layer.output[0])
                                      })
        if num_opt > 1 and node.layer.output[1] != '':
            node.fluid_code.add_layer('unsqueeze',
                                      inputs=var_y00,
                                      output=node.layer.output[1],
                                      param_attr={
                                          'axes': [1, 1],
                                          'name': string(node.layer.output[1])
                                      })
