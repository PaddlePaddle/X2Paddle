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
from x2paddle.core.util import *
from x2paddle.core.fluid_code import Layer
from x2paddle.core.fluid_code import FluidCode
from x2paddle.decoder.onnx_decoder import ONNXGraph, ONNXGraphNode, ONNXGraphDataNode
from x2paddle.op_mapper.onnx_directly_map import default_op_mapping_field_values
from x2paddle.op_mapper.onnx_directly_map import default_op_mapping
from x2paddle.op_mapper.onnx_directly_map import default_ioa_constraint
import numpy as np
import logging as _logging
from collections import OrderedDict as _dict

_logger = _logging.getLogger(__name__)


def _const_weight_or_none(node):
    if 'Constant' in node.layer_name:
        return val.value
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
    def __init__(self, decoder):
        super(ONNXOpMapper, self).__init__()
        self.decoder = decoder
        self.graph = decoder.onnx_graph
        self.input_shapes = []
        self.weights = dict()
        self.omit_nodes = list()

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

    def op_checker(self):
        unsupported_ops = set()
        for node_name in self.graph.topo_sort:
            node = self.graph.get_node(node_name)
            op = node.layer_type
            if not hasattr(self, op) and op not in default_op_mapping:
                unsupported_ops.add(op)
        if len(unsupported_ops) == 0:
            return True
        else:
            print("There are {} ops not supported yet, list as below".format(
                len(unsupported_ops)))
            for op in unsupported_ops:
                print(op)
            return False

    def directly_map(self, node, *args, name='', **kwargs):
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
        val_inps = inputs if input_perm is None else list(
            map(lambda i: inputs[i], input_perm))
        val_outs = outputs if output_perm is None else list(
            map(lambda i: outputs[i], output_perm))
        attr = fluid_attrs
        if fluid_op not in ['shape', 'gather']:
            attr['name'] = string(node.layer_name)
        node.fluid_code.add_layer(fluid_op,
                                  inputs=', '.join(val_inps),
                                  output=val_outs[0],
                                  param_attr=attr)

    def place_holder(self, node):
        self.input_shapes.append(node.out_shapes)
        attr = {
            "dtype": string(node.dtype),
            "shape": node.out_shapes,
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
        shape = node.out_shapes

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

    def Pad(self, node, op_independent=True):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        pads = node.get_attr('pads')
        mode = node.get_attr('mode', 'constant')
        value = node.get_attr('value', 0.)
        data_shape = val_x.out_shapes
        output_shape = node.out_shapes
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
            assert mode == 'constant', 'mode {} is supported only in pad2d'.format(
                mode)
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
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        axes = node.get_attr('axes')
        attr = {'axes': axes, 'name': string(node.layer_name)}
        node.fluid_code.add_layer('unsqueeze',
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
            shape = val_output.out_shapes
        if shape is None:
            shape = list(value.shape)
            _logger.warning(
                'in (Constant -> %s): '
                'attribute "shape" of %s not inferred, '
                'using value as 1-D tensor may lead to fails',
                val_output.layer_name, val_output.layer_name)

        value = value.tolist()
        if len(value) == 1:  # scalar
            shape = [1]
            value = value[0]
            if dtype.name == 'int64':
                dtype = 'int32'
            attr = {'shape': shape, 'dtype': string(dtype), 'value': value}
            node.fluid_code.add_layer('fill_constant',
                                      inputs=None,
                                      output=node,
                                      param_attr=attr)

    def Resize(self, node):
        # I/O
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        val_scales = self.graph.get_node(node.layer.input[1], copy=True)
        val_y, = self.graph.get_node(node.layer.output[0], copy=True)

        out_shape_ = val_y.out_shapes
        if out_shape_ is not None:
            assert len(out_shape_) == 4, 'only 4-D Tensor as X and Y supported'
            out_shape_ = out_shape_[2:]
        scales = _const_weight_or_none(val_scales)
        if scales is not None:
            assert len(scales) == 4, 'only 4-D Tensor as X and Y supported'
            assert scales[0] == 1 and scales[
                1] == 1, 'only scale on (NC)HW supported'
            assert scales[2] == scales[
                3], 'only aspect-ratio-invariant scale supported'
        scale = scales[2] if scales else None
        if scale is None:
            assert out_shape_, 'neither scales nor output shape is available'
            out_shape = out_shape_
        else:
            out_shape = None
            if out_shape_ is None:
                in_shape = val_x.out_shapes
                assert in_shape is not None, 'out_shape required but not inferrable'
                assert len(
                    in_shape) == 4, 'only 4-D Tensor as X and Y supported'
                out_shape_ = [in_shape[2] * scale, in_shape[3] * scale]

        mode = node.get_attr('mode', 'nearest')
        fluid_op = 'resize_{}'.format(mode)
        name_attr = ', name={}'.format(repr(name)) if name else ''

        attr = {
            'scale': scale,
            'out_shape': out_shape,
            'name': string(node.layer_name)
        }
        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def ConstantOfShape(self, node):
        val_shape = self.graph.get_node(node.layer.input[0], copy=True)

        shape = _const_weight_or_none(val_shape)

        if shape is None:
            shape = node.out_shapes

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
        val_input = self.graph.get_node(node.layer.input[0], copy=True)
        var_outs = [val for val in node.layer.input]

        fluid_op = 'split'
        split = node.get_attr['split']
        axis = node.get_attr('axis', 0)
        attr = {'split': split, 'axis': axis, 'name': string(node.layer_name)}
        # generation
        node.fluid_code.add_layer('split',
                                  inputs=val_input,
                                  output=var_outs,
                                  param_attr=attr)

    def Reshape(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        val_shape = self.graph.get_node(node.layer.input[1], copy=True)
        val_reshaped = self.graph.get_node(node.layer.output[0], copy=True)
        shape = None
        if isinstance(val_shape, ONNXGraphDataNode):
            self.omit_nodes.append(val_shape.layer_name)

        # catch dynamic graph shape
        if isinstance(val_shape, ONNXGraphNode):
            shape = self.decoder.get_dynamic_shape_from_caffe2(
                val_shape.layer_name, self.input_shapes)
        if shape is None:
            shape = val_reshaped.out_shapes

        shape_dtype = val_shape.dtype

        if shape_dtype is None:
            _logger.warning(
                'in op %s(%s -> Reshape -> %s): '
                'dtype of input "shape" not inferred, int32 assumed',
                node.layer_name, val_x.layer_name, val_reshaped.layer_name)
            shape_dtype = _np.dtype('int32')
        if shape is None:
            shape = [1, -1]
            _logger.warning(
                'in %s(%s -> Reshape -> %s): '
                'input "shape" not inferred, use [1, -1] as dummy value, '
                'the behavior of Paddle fluid maybe undefined', node.layer_name,
                val_x.layer_name, val_reshaped.layer_name)
        attr = {'shape': shape, 'name': string(node.layer_name)}

        node.fluid_code.add_layer('reshape',
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Cast(self, node):
        val_input = self.graph.get_node(node.layer.input[0], copy=True)
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
        val_x = self.graph.get_node(node.layer.input[0], copy=True)

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

        input_shape = val_x.out_shapes
        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
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
            ipt = self.graph.get_node(node.layer.input[i], copy=True)
            if isinstance(ipt, str):
                inputs.append(ipt)
            else:
                inputs.append(ipt.layer_name)
        axis = node.get_attr('axis')
        attr = {'axis': axis}
        node.fluid_code.add_layer('concat',
                                  inputs='[' + ', '.join(inputs) + ']',
                                  output=node,
                                  param_attr=attr)

    def Flatten(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        axis = node.get_attr('axis', 1)
        attr = {"axis": str(axis), "name": string(node.layer_name)}
        node.fluid_code.add_layer('flatten',
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Gemm(self, node):
        val_a = self.graph.get_node(node.layer.input[0], copy=True)
        val_b = self.graph.get_node(node.layer.input[1], copy=True)
        val_c = self.graph.get_node(node.layer.input[2], copy=True)

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

    def Add(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        val_y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {
            "x": val_x,
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
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        val_y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {"x": val_x, "y": val_y}
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer("matmul",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)

    def BatchNormalization(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        val_scale = self.graph.get_node(node.layer.input[1], copy=True)
        val_b = self.graph.get_node(node.layer.input[2], copy=True)
        val_mean = self.graph.get_node(node.layer.input[3], copy=True)
        val_var = self.graph.get_node(node.layer.input[4], copy=True)

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

    def Softmax(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer("softmax",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Transpose(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        perm = node.get_attr('perm')
        attr = {'perm': perm, "name": string(node.layer_name)}
        node.fluid_code.add_layer("transpose",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Div(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        val_y = self.graph.get_node(node.layer.input[1], copy=True)
        inputs = {'x': val_x, 'y': val_y}
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer("elementwise_div",
                                  inputs=inputs,
                                  output=node,
                                  param_attr=attr)

    def Relu(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        attr = {"name": string(node.layer_name)}
        node.fluid_code.add_layer("relu",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def PRelu(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        val_slope = self.graph.get_node(node.layer.input[1], copy=True)
        attr = {"name": string(node.layer_name), "mode": string('channel')}

        if isinstance(val_slope, str):
            attr["param_attr"] = string(val_slope.layer_name)
        else:
            attr["param_attr"] = string(val_slope.layer_name)
        node.fluid_code.add_layer("prelu",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Squeeze(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        squeeze_dims = node.get_attr('squeeze_dims')
        attr = {'axes': squeeze_dims, "name": string(node.layer_name)}
        node.fluid_code.add_layer("squeeze",
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Identity(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        node.fluid_code.add_layer("assign", inputs=val_x, output=node)

    def MaxPool(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)

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

        input_shape = val_x.out_shapes
        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
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

    def GlobalAveragePool(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)
        input_shape = val_x.out_shapes
        output_shape = val_y.out_shapes
        assert input_shape is not None or output_shape is not None, 'poolnd not inferred'  # N
        if input_shape:
            poolnd = len(input_shape) - 2  # NC...
        elif output_shape:
            poolnd = len(output_shape) - 2  # NC...
        assert 2 <= poolnd <= 3, 'only pool2d and pool3d is supported'
        fluid_op = 'pool{}d'.format(poolnd)
        attr = {
            "pool_type": string("avg"),
            "global_pooling": True,
            "name": string(node.layer_name)
        }
        node.fluid_code.add_layer(fluid_op,
                                  inputs=val_x,
                                  output=node,
                                  param_attr=attr)

    def Conv(self, node):
        val_x = self.graph.get_node(node.layer.input[0], copy=True)
        val_w = self.graph.get_node(node.layer.input[1], copy=True)
        val_y = self.graph.get_node(node.layer.output[0], copy=True)

        self.omit_nodes.append(val_w.layer_name)
        input_shape = val_x.out_shapes

        has_bias = len(node.layer.input) == 3
        if has_bias:
            val_b = self.graph.get_node(node.layer.input[2], copy=True)
            self.omit_nodes.append(val_b.layer_name)
        auto_pad = node.get_attr('auto_pad', 'NOTSET')

        kernel_shape = val_w.out_shapes[2:]  # OI...
        assert kernel_shape == node.get_attr(
            'kernel_shape'), 'kernel_shape in attr unmatches value_info'  # HW
        convnd = len(kernel_shape)
        assert 2 <= convnd <= 3, 'only conv2d and conv3d is supported'
        num_out_channels = val_w.out_shapes[0]  # OI...
        fluid_op = 'conv{}d'.format(convnd)

        num_groups = node.get_attr('group', 1)
        strides = node.get_attr('strides', [1] * convnd)  # optional
        dilations = node.get_attr('dilations', [1] * convnd)  # optional
        pads = node.get_attr('pads', [0] * (convnd * 2))  # optional

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
