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

from x2paddle.decoder.onnx_decoder import ONNXGraph, ONNXGraphNode, ONNXGraphDataNode
import numpy as np
import sympy


def handle_negative_axis(axis, rank):
    return axis if axis >= 0 else axis + rank


class ShapeInference():
    def __init__(self, decoder, auto_merge=False):
        self.decoder = decoder
        self.fluid_data = {}
        self.suggested_merge_ = {}
        self.symbolic_dims_ = {}
        self.auto_merge_ = auto_merge
        self.dispatcher = {
            # activation ops
            'Relu': self.activation_ops,
            'LeakyRelu': self.activation_ops,
            'Elu': self.activation_ops,
            'ThresholdRelu': self.activation_ops,
            'Prelu': self.activation_ops,
            'Tanh': self.activation_ops,
            'Sigmoid': self.activation_ops,
            'Softplus': self.activation_ops,
            'Softsign': self.activation_ops,
            'HardSigmoid': self.activation_ops,
            'Shrink': self.activation_ops,
            'Exp': self.activation_ops,
            'Clip': self.activation_ops,

            # elementwise ops
            'Add': self.elementwise_ops,
            'Div': self.elementwise_ops,
            'Sub': self.elementwise_ops,
            'Mul': self.elementwise_ops,
            'Pow': self.elementwise_ops,
            'Sqrt': self.elementwise_ops,
            'Softmax': self.elementwise_ops,
            'Constant': self.constant,
            'AveragePool': self.pool,
            'MaxPool': self.pool,
            'Cast': self.cast,
            'Conv': self.conv,
            'BatchNormalization': self.batch_norm,
            'Pad': self.pad,
            'Gather': self.gather,
            'Split': self.split,
            'Transpose': self.transpose,
            'Reshape': self.reshape,
            'MatMul': self.matmul,
            'Squeeze': self.squeeze,
            'Unsqueeze': self.unsqueeze,
            'Concat': self.concat,
        }
        self.run_ = True
        self.suggested_merge_ = {}
        self.symbolic_dims_ = {}
        self.input_symbols_ = {}

    def __call__(self):
        """
        run shape inference
        """
        nodes = self.decoder.model.graph.node
        node_map = self.decoder.onnx_graph.node_map
        value_infos = self.decoder.onnx_graph.value_infos
        onnx_model = self.decoder.model
        #self._apply_suggested_merge(graph_input_only=True)
        for layer in nodes:
            node = node_map[layer.name]
            for opt in layer.output:
                if opt in value_infos:
                    value_info = value_infos[opt]
                    #if len(value_info['shape']) == 0 or value_info[
                    #        'dtype'] is None or 0 in value_info['shape']:
                    #    #TODO add node shape inference
                    #    if self.is_support_inference(node):
                    #        op_infer = self.dispatcher[node.layer_type]
                    #        #shapes = op_infer(node)
                    #        print(node.layer_name + ': ')
                    #        print(node.layer_type + ': ')
                    #else:
                    #   print(node.layer_name)
                    node.dtype = value_info['dtype']
                    node.out_shapes.append(value_info['shape'])
                else:
                    #TODO add node shape inference
                    if self.is_support_inference(node):
                        op_infer = self.dispatcher[node.layer_type]
                        #shapes = op_infer(node)
                        #print(node.layer_name + ': ')
                        #print(node.layer_type + ': ')

    def get_input_node(self, node, idx, copy=False):
        return self.decoder.onnx_graph.get_input_node(node, idx=idx, copy=copy)

    def get_fluid_data(self, node, return_ndarray=False):
        data = None
        if node.layer_name in self.fluid_data:
            data = self.fluid_data[node.layer_name]
        elif isinstance(node, ONNXGraphDataNode):
            data = node.weight
        elif isinstance(node, ONNXGraphNode):
            data = node.value
        if return_ndarray:
            return data
        else:
            return data.tolist()

    def is_support_inference(self, node):
        if node.layer_type not in self.dispatcher:
            print(
                "[WARNNING] Shape inference not support Node[{}](op type: {}) ".
                format(node.layer_name, node.layer_type))
            return False
        return True

    def _try_get_value(self, node, idx):
        if idx >= len(node.inputs):
            return None
        return self.get_input_node(node, idx=idx, return_ndarray=True)

    def _get_int_values(self, node, broadcast=False):
        values = [self._try_get_value(node, i) for i in range(len(node.input))]
        if all([v is not None for v in values]):
            # some shape compute is in floating point, cast to int for sympy
            for i, v in enumerate(values):
                if type(v) != np.ndarray:
                    continue
                if len(v.shape) > 1:
                    new_v = None  # ignore value for rank > 1
                elif len(v.shape) == 0:
                    new_v = int(np.asscalar(v))
                else:
                    assert len(v.shape) == 1
                    new_v = [int(vv) for vv in v]
                values[i] = new_v
        values_len = [len(v) if type(v) == list else 0 for v in values]
        max_len = max(values_len)
        if max_len >= 1 and broadcast:
            # broadcast
            for i, v in enumerate(values):
                if v is None:
                    continue  # don't broadcast if value is unknown
                if type(v) == list:
                    if len(v) < max_len:
                        values[i] = v * max_len
                    else:
                        assert len(v) == max_len
                else:
                    values[i] = [v] * max_len
        return values

    def _compute_on_sympy_data(self, node, op_func):
        assert len(node.outputs) == 1
        values = self._get_int_values(node, broadcast=True)
        if all([v is not None for v in values]):
            is_list = [type(v) == list for v in values]
            as_list = any(is_list)
            if as_list:
                data = [op_func(vs) for vs in zip(*values)]
                self.fluid_data[node.layer_name] = data
                node.out_shapes.append(data.shape)
                print('*' * 10, data)
            else:
                data = op_func(values)
                self.fluid_data[node.layer_name] = data
                print('*' * 10, data)
                node.out_shapes.append(data.shape)

    def _pass_on_sympy_data(self, node):
        assert len(node.inputs) == 1 or node.layer_type == 'Reshape'
        self._compute_on_sympy_data(node, lambda x: x[0])

    def _get_sympy_shape(self, node, idx):
        sympy_shape = []
        for d in self._get_shape(node, idx):
            if type(d) == str:
                sympy_shape.append(self.symbolic_dims_[d] if d in
                                   self.symbolic_dims_ else sympy.Symbol(
                                       d, integer=True))
            else:
                assert None != d
                sympy_shape.append(d)
        return sympy_shape

    def _check_merged_dims(self, dims, allow_broadcast=True):
        if allow_broadcast:
            dims = [d for d in dims if not (is_literal(d) and int(d) <= 1)]
        if not all([d == dims[0] for d in dims]):
            self._add_suggested_merge(dims, apply=True)

    def check_specific_shape(self, input_node, output_node, shape):
        if -1 in input_node.out_shapes[0]:
            assert "Shape inference failed, when calculate output_node[{}]'s  \
            shape need specific shape, but got input_node[{}]'s shape: {}".format(
                output_node.layer_name, input_node.layer_name,
                input_node.out_shapes[0])

    def _add_suggested_merge(self, symbols, apply=False):
        assert all([(type(s) == str and s in self.symbolic_dims_) or
                    is_literal(s) for s in symbols])
        symbols = set(symbols)
        for k, v in self.suggested_merge_.items():
            if k in symbols:
                symbols.remove(k)
                symbols.add(v)
        map_to = None
        # if there is literal, map to it first
        for s in symbols:
            if is_literal(s):
                map_to = s
                break
        # when no literals, map to input symbolic dims, then existing symbolic dims
        if map_to is None:
            for s in symbols:
                if s in self.input_symbols_:
                    map_to = s
                    break
        if map_to is None:
            for s in symbols:
                if type(self.symbolic_dims_[s]) == sympy.Symbol:
                    map_to = s
                    break
        # when nothing to map to, use the shorter one
        if map_to is None:
            if self.verbose_ > 0:
                print(
                    'Potential unsafe merge between symbolic expressions: ({})'.
                    format(','.join(symbols)))
            symbols_list = list(symbols)
            lens = [len(s) for s in symbols_list]
            map_to = symbols_list[lens.index(min(lens))]
            symbols.remove(map_to)

    def _merge_symbols(self, dims):
        if not all([type(d) == str for d in dims]):
            if self.auto_merge_:
                assert len(
                    dims
                ) == 2  # only allow symbol->int merge in binary ops for now
                is_int = [is_literal(d) for d in dims]
                if sum(is_int) == 1:
                    int_dim = is_int.index(1)
                    if self.verbose_ > 0:
                        print('dim {} has been merged with value {}'.format(
                            dims[1 - int_dim], dims[int_dim]))
                    self._check_merged_dims(dims, allow_broadcast=False)
                    return dims[int_dim]
                else:
                    if self.verbose_ > 0:
                        print('dim {} has been mergd with dim {}'.format(dims[
                            0], dims[1]))
                    return dims[0]
            else:
                return None
        if all([d == dims[0] for d in dims]):
            return dims[0]
        merged = [
            self.suggested_merge_[d] if d in self.suggested_merge_ else d
            for d in dims
        ]
        if all([d == merged[0] for d in merged]):
            assert merged[0] in self.symbolic_dims_
            return merged[0]
        else:
            return None

    # broadcast from right to left, and merge symbolic dims if needed
    def _broadcast_shapes(self, shape1, shape2):
        new_shape = []
        rank1 = len(shape1)
        rank2 = len(shape2)
        new_rank = max(rank1, rank2)
        for i in range(new_rank):
            dim1 = shape1[rank1 - 1 - i] if i < rank1 else 1
            dim2 = shape2[rank2 - 1 - i] if i < rank2 else 1
            if dim1 == 1 or dim1 == dim2:
                new_dim = dim2
            elif dim2 == 1:
                new_dim = dim1
            else:
                new_dim = self._merge_symbols([dim1, dim2])
                if not new_dim:
                    # warning about unsupported broadcast when not auto merge
                    # note that auto merge has the risk of incorrectly merge symbols while one of them being 1
                    # for example, 'a' = 1, 'b' = 5 at runtime is valid broadcasting, but with auto merge 'a' == 'b'
                    if self.auto_merge_:
                        self._add_suggested_merge([dim1, dim2], apply=True)
                    else:
                        print('unsupported broadcast between ' + str(dim1) + ' '
                              + str(dim2))
            new_shape = [new_dim] + new_shape
        return new_shape

    def _apply_suggested_merge(self, graph_input_only=False):
        if not self.suggested_merge_:
            return
        for i in list(self.decoder.model.graph.input) + (
            [] if graph_input_only else
                list(self.decoder.model.graph.value_info)):
            for d in i.type.tensor_type.shape.dim:
                if d.dim_param in self.suggested_merge_:
                    v = self.suggested_merge_[d.dim_param]
                    if is_literal(v):
                        d.dim_value = int(v)
                    else:
                        d.dim_param = v

    def _add_suggested_merge(self, symbols, apply=False):
        assert all([(type(s) == str and s in self.symbolic_dims_) or
                    is_literal(s) for s in symbols])
        symbols = set(symbols)
        for k, v in self.suggested_merge_.items():
            if k in symbols:
                symbols.remove(k)
                symbols.add(v)
        map_to = None
        # if there is literal, map to it first
        for s in symbols:
            if is_literal(s):
                map_to = s
                break
        # when no literals, map to input symbolic dims, then existing symbolic dims
        if map_to is None:
            for s in symbols:
                if s in self.input_symbols_:
                    map_to = s
                    break
        if map_to is None:
            for s in symbols:
                if type(self.symbolic_dims_[s]) == sympy.Symbol:
                    map_to = s
                    break
        # when nothing to map to, use the shorter one
        if map_to is None:
            if self.verbose_ > 0:
                print(
                    'Potential unsafe merge between symbolic expressions: ({})'.
                    format(','.join(symbols)))
            symbols_list = list(symbols)
            lens = [len(s) for s in symbols_list]
            map_to = symbols_list[lens.index(min(lens))]
            symbols.remove(map_to)

        for s in symbols:
            if s == map_to:
                continue
            if is_literal(map_to) and is_literal(s):
                assert int(map_to) == int(s)
            self.suggested_merge_[s] = int(map_to) if is_literal(
                map_to) else map_to
            for k, v in self.suggested_merge_.items():
                if v == s:
                    self.suggested_merge_[k] = map_to
        if apply and self.auto_merge_:
            self._apply_suggested_merge()

    def pool_conv_ops(self, node):
        fluid_shape = self.get_input_node(node, idx=0).out_shapes[0]
        if len(node.inputs) > 1:
            W_shape = self.get_input_node(node, idx=1).out_shapes[0]
            rank = len(W_shape) - 2  # number of spatial axes
            kernel_shape = W_shape[-rank:]
            sympy_shape[1] = W_shape[0]
        else:
            W_shape = None
            kernel_shape = node.get_attr('kernel_shape')
            rank = len(kernel_shape)
        dilations = node.get_attr('dilations', [1] * rank)
        strides = node.get_attr('strides', [1] * rank)
        pads = node.get_attr('pads')
        effective_kernel_shape = [(k - 1) * d + 1
                                  for k, d in zip(kernel_shape, dilations)]
        if pads is None:
            pads = [0] * (2 * rank)
            auto_pad = node.get_attr('auto_pad', b'NOTSET').decode('utf-8')
            if auto_pad != 'VALID' and auto_pad != 'NOTSET':
                try:
                    residual = [
                        sympy.Mod(d, s)
                        for d, s in zip(fluid_shape[-rank:], strides)
                    ]
                    total_pads = [
                        max(0, (k - s) if r == 0 else (k - r))
                        for k, s, r in zip(effective_kernel_shape, strides,
                                           residual)
                    ]
                except TypeError:  # sympy may throw TypeError: cannot determine truth value of Relational
                    total_pads = [
                        max(0, (k - s))
                        for k, s in zip(effective_kernel_shape, strides)
                    ]  # assuming no residual if sympy throws error
            elif auto_pad == 'VALID':
                total_pads = []
            else:
                total_pads = [0] * rank
        else:
            assert len(pads) == 2 * rank
            total_pads = [p1 + p2 for p1, p2 in zip(pads[:rank], pads[rank:])]
        ceil_mode = node.get_attr('ceil_mode', 0)
        for i in range(rank):
            effective_input_size = fluid_shape[-rank + i]
            if len(total_pads) > 0:
                effective_input_size = effective_input_size + total_pads[i]
            if ceil_mode:
                strided_kernel_positions = sympy.ceiling(
                    (effective_input_size - effective_kernel_shape[i]) /
                    strides[i])
            else:
                strided_kernel_positions = (
                    effective_input_size - effective_kernel_shape[i]
                ) // strides[i]
            fluid_shape[-rank + i] = strided_kernel_positions + 1
        node.out_shapes.append(fluid_shape)
        return fluid_shape

    def cast(self, node):
        fluid_shape = self.get_input_node(node, idx=0).out_shape[0]
        node.out_shapes.append(fluid_shape)
        return fluid_shape

    def pool(self, node):
        return self.conv_pool_ops(node)

    def conv(self, node):
        return self.conv_pool_ops(node)

    def batch_norm(self, node):
        fluid_shape = self.get_input_node(node, idx=0).out_shapes[0]
        node.out_shapes.append(fluid_shape)
        return fluid_shape

    def activation_ops(self, node):
        fluid_shape = self.get_input_node(node, idx=0).out_shapes[0]
        node.out_shapes.append(fluid_shape)
        return fluid_shape

    def elementwise_ops(self, node):
        fluid_shape = self.get_input_node(node, idx=0).out_shapes[0]
        node.out_shapes.append(fluid_shape)
        return fluid_shape

    def pad(self, node):
        fluid_shape = self.get_input_node(node, idx=0).out_shapes[0]
        # op_set <= 10
        pads = node.get_attr('pads')

        rank = len(fluid_shape)
        fluid_shape = [
            d + pad_up + pad_down
            for d, pad_up, pad_down in zip(fluid_shape, pads[:rank], pads[
                rank:])
        ]
        node.out_shapes.append(fluid_shape)
        return fluid_shape

    def gather(self, node):
        fluid_shape = self.get_input_node(node, idx=0).out_shapes[0]
        axis = handle_negative_axis(node.get_attr('axis', 0), len(fluid_shape))
        indices_shape = self.get_input_node(node, idx=1).out_shapes[0]
        fluid_shape = fluid_shape[:axis] + list(indices_shape) + fluid_shape[
            axis + 1:]
        input = self.get_input_node(node, 0)
        if input.layer_name in self.fluid_data:
            assert 0 == axis  # only handle 1D sympy compute
            idx = self.get_fluid_date(indices_shape)
            data = self.fluid_data[input.layer_name]
            if type(data) == list:
                if type(idx) == np.ndarray and len(idx.shape) == 1:
                    self.fluid_data[
                        node.layer_name] = [data[int(i)] for i in idx]
                else:
                    self.fluid_data[node.layer_name] = data[int(idx)]
            else:
                assert idx == 0
                self.fluid_data[node.layer_name] = data

        node.out_shapes.append(fluid_shape)
        return fluid_shape

    def constant(self, node):
        if isinstance(node, ONNXGraphNode):
            fluid_shape = node.value.shape
        else:
            fluid_shape = node.weight.shape

        node.out_shapes.append(fluid_shape)
        return fluid_shape

    def split(self, node):
        fluid_shape = self.get_input_node(node, idx=0).out_shapes[0]
        axis = handle_negative_axis(node.get_attr('axis', 0), len(fluid_shape))
        split = node.get_attr('split')

        if not split:
            num_outputs = len(node.outputs)
            split = [fluid_shape[axis] /
                     sympy.Integer(num_outputs)] * num_outputs
        else:
            split = [sympy.Integer(s) for s in split]
        shapes = []
        for i_o in range(len(split)):
            shape = fluid_shape[:axis] + [split[i_o]] + fluid_shape[axis + 1:]
            shapes.append(shape)
        node.out_shapes += shapes

        return shapes

    def shape(self, node):
        fluid_shape = self.get_input_node(node, idx=0).out_shapes[0]
        fluid_shape = [len(fluid_shape), ]
        node.out_shapes.append(fluid_shape)
        self.fluid_data[node.layer_name] = np.array(fluid_shape)
        return fluid_shape

    def transpose(self, node):
        fluid_shape = self.get_input_node(node, idx=0).out_shapes[0]
        perm = node.get_attr('perm')
        fulid_shape = np.array(fluid_shape)[perm].tolist()
        node.out_shapes.append(fluid_shape)
        return fluid_shape

    def reshape(self, node):
        shape = self.get_input_node(node, idx=1)
        shape_data = self.get_fluid_data(shape)
        if shape_data is not None:
            if -1 in shape_data:
                fluid_shape = self.get_input_node(node, idx=0).out_shapes[0]
                print(fluid_shape)
                index = shape_data.index(-1)
                total_elements = 1
                for dim in fluid_shape:
                    total_elements *= dim
                part_elements = 1
                for dim in shape_data:
                    if dim != -1:
                        part_elements *= dim
                shape_data[index] = total_elements // part_elements
            node.out_shapes.append(shape_data)
        else:
            pass
        return shape_data

    def matmul(self, node):
        x_shape = self.get_input_node(node, idx=0).out_shapes[0]
        y_shape = self.get_input_node(node, idx=1).out_shapes[0]
        x_rank = len(x_shape)
        y_rank = len(y_shape)
        if x_rank == 1 and y_rank == 1:
            new_shape = []
        elif x_rank == 1:
            y_reduce_dim = -2
            new_shape = x_shape[:y_reduce_dim] + [x_shape[-1]]
        elif y_rank == 1:
            x_reduce_dim = -1
            new_shape = x_shape[:x_reduce_dim]
        else:
            x_reduce_dim = -1
            y_reduce_dim = -2
            new_shape = self._broadcast_shapes(
                x_shape[:-2], y_shape[:-2]) + [x_shape[-2]] + [y_shape[-1]]
        node.out_shapes.append(new_shape)
        return new_shape

    def squeeze(self, node):
        self._pass_on_sympy_data(node)

    def unsqueeze(self, node):
        self._pass_on_sympy_data(node)

    def concat(self, node):
        if any([i in self.fluid_data for i in node.inputs]):
            values = self._get_int_values(node)
            if all([v is not None for v in values]):
                assert 0 == get_attribute(node, 'axis')
                self.fluid_data[node.layer_name] = []
                for i in range(len(node.input)):
                    value = values[i]
                    if type(value) == list:
                        self.fluid_data[node.layer_name].extend(value)
                    else:
                        self.fluid_data[node.layer_name].append(value)
