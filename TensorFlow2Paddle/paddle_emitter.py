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

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util
from six import string_types as _string_types


class PaddleEmitter(object):
    skip_op = set(['const', 'identity'])
    dtype_map = {1: "float32", 3: "int32", 9: "int64"}

    def __init__(self, graph):
        self.graph = graph
        self.body_code = ""
        self.tab = " " * 4

    @staticmethod
    def tensor_shape_to_list(shapes):
        if isinstance(shapes, attr_value_pb2.AttrValue):
            return [dim.size for dim in shapes.shape.dim]
        else:
            ret = []
            for shape in shapes:
                this_one = [dim.size for dim in shape.dim]
                ret.append(this_one)
            return ret

    @property
    def header_code(self):
        code = ["import paddle.fluid as fluid", "", "def KitModel():"]
        return code

    def add_codes(self, indent, codes):
        if isinstance(codes, _string_types):
            codes = codes.strip().split("\n")
        for code in codes:
            self.body_code += (self.tab * indent) + code + "\n"

    def gen_code(self):
        self.add_codes(0, self.header_code)

        for node in self.graph.topological_sort:
            current_node = self.graph.get_node(node)
            op = current_node.type
            if op in self.skip_op:
                continue
            if hasattr(self, "emit_" + op):
                func = getattr(self, "emit_" + op)
                codes = func(current_node)
                if not isinstance(codes, list):
                    codes = [codes]
                self.graph.get_node(node).codes = codes
            else:
                raise Exception("Unknow node op: {}".format(op))

        for node in self.graph.topological_sort:
            codes = self.graph.get_node(node).codes
            self.add_codes(1, codes)

        outs = []
        for node in self.graph.output_nodes:
            outs.append(self.graph.get_node(node).ref_name)
        self.add_codes(1, "return {}".format(", ".join(outs)))

        return self.body_code

    def gen_weight(self, weight_dict, dirname):
        import struct
        import framework_pb2 as framework
        import numpy

        for var_name, var in weight_dict.items():
            if var_name not in self.graph.node_map:
                continue
            shape = var.shape
            paddle_var_name = self.graph.get_node(var_name).ref_name
            paddle_var = var
            dataformat = self.graph.get_node(var_name).dataformat

            filew = open(dirname + '/' + paddle_var_name, 'wb')
            filew.write(struct.pack('i', 0))
            filew.write(struct.pack('L', 0))
            filew.write(struct.pack('i', 0))
            tensor_desc = framework.VarType.TensorDesc()
            tensor_desc.data_type = framework.VarType.FP32

            if len(shape) == 4 and dataformat == "NHWC":
                paddle_var = numpy.transpose(var, (3, 2, 0, 1))
                shape = paddle_var.shape

            tensor_desc.dims.extend(shape)
            desc_size = tensor_desc.ByteSize()
            filew.write(struct.pack('i', desc_size))
            filew.write(tensor_desc.SerializeToString())

            tensor_size = reduce(lambda x, y: x * y, shape)
            paddle_var = paddle_var.flatten()
            for i in range(0, tensor_size):
                filew.write(struct.pack('f', paddle_var[i]))
            filew.close()

    def emit_variablev2(self, node):
        shape = self.tensor_shape_to_list(node.get_attr('_output_shapes'))[0]
        if node.dataformat == 'NHWC' and len(shape) == 4:
            shape = [shape[3], shape[2], shape[0], shape[1]]

        dtype = node.get_attr('dtype')
        if dtype in self.dtype_map:
            dtype = self.dtype_map[dtype]
        else:
            raise Exception('Unknow dtype : {}'.format(dtype))

        code = ["# variable[{}]:\t{}".format(node.ref_name, node.name)]
        code.append(
            '{} = fluid.layers.create_parameter(name=\'{}\', shape={}, dtype=\'{}\')'
            .format(node.ref_name, node.ref_name, shape, dtype))
        return code

    def emit_placeholder(self, node):
        shape = self.tensor_shape_to_list(node.get_attr('shape'))[0]

        if node.dataformat == 'NHWC' and len(shape) == 4:
            shape = [shape[0], shape[3], shape[1], shape[2]]
        if shape[0] < 0:
            shape = shape[1:]

        dtype = node.get_attr('dtype')
        if dtype in self.dtype_map:
            dtype = self.dtype_map[dtype]
        else:
            raise Exception('Unknow dtype : {}'.format(dtype))

        code = ["# placeholder[{}]:\t{}".format(node.ref_name, node.name)]
        code.append(
            '{} = fluid.layers.data(name=\'{}\', shape={}, dtype=\'{}\')'.
            format(node.ref_name, node.ref_name, shape, dtype))
        return code

    def emit_conv2d(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        kernel = self.graph.get_node(node.layer.input[1])

        dataformat = node.dataformat
        padding_mode = node.get_attr('padding')
        strides = node.get_attr('strides')[1:3]
        k_shape = self.tensor_shape_to_list(
            kernel.get_attr('_output_shapes'))[0]

        kernel_num, channel, height, width = k_shape
        if dataformat == "NHWC":
            height, width, channle, kernel_num = k_shape

        padding = [0, 0]
        if padding_mode == 'SAME':
            padding = map(int, [(height - 1) / 2, (width - 1) / 2])

        code = '{} = fluid.layers.conv2d({}, {}, {}, padding={}, stride={}, param_attr=\'{}\', bias_attr=False)'.format(
            node.ref_name, inputs.ref_name, kernel_num, [height, width],
            padding, strides, kernel.ref_name)

        return code

    def emit_biasadd(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        bias = self.graph.get_node(node.layer.input[1])
        # TODO more validations
        axis = 1
        code = '{} = fluid.layers.elementwise_add({}, {}, axis={})'.format(
            node.ref_name, inputs.ref_name, bias.ref_name, axis)
        return code

    def emit_relu(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        code = '{} = fluid.layers.relu({})'.format(node.ref_name,
                                                   inputs.ref_name)
        return code

    def emit_maxpool(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        padding_mode = node.get_attr('padding')
        strides = node.get_attr('strides')[1:3]
        pool_size = node.get_attr('ksize')[1:3]
        padding = [0, 0]
        if padding_mode == 'SAME':
            pad_0 = (pool_size[0] - 1) / 2
            pad_1 = (pool_size[1] - 1) / 2
            padding = [0, pad_0 * 2, 0, pad_1 * 2]
            code = [
                'pad_net = fluid.layers.pad2d({}, paddings={})'.format(
                    inputs.ref_name, padding)
            ]
            code.append(
                '{} = fluid.layers.pool2d(pad_net, {}, \'max\', {})'.format(
                    node.ref_name, pool_size, strides))
        else:
            code = '{} = fluid.layers.pool2d({}, {}, \'max\', {})'.format(
                node.ref_name, inputs.ref_name, pool_size, strides)
        return code

    def emit_pad(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        padding = self.graph.get_node(node.layer.input[1])
        assert padding.type == 'const'
        padding = padding.layer.attr['value'].tensor
        padding = tensor_util.MakeNdarray(padding)
        if node.dataformat == "NHWC" and padding.shape[0] == 4:
            padding = padding[[0, 3, 1, 2]]
        code = '{} = fluid.layers.pad({}, {})'.format(node.ref_name,
                                                      inputs.ref_name,
                                                      list(padding.flatten()))
        return code

    def emit_fusedbatchnorm(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        gamma = self.graph.get_node(node.layer.input[1])
        beta = self.graph.get_node(node.layer.input[2])
        mv_mean = self.graph.get_node(node.layer.input[3])
        mv_variance = self.graph.get_node(node.layer.input[4])

        is_training = node.get_attr("is_training")
        if is_training:
            raise Exception(
                "FusedBatchNorm: is_training=True, not support yet, please set is_training=False in your tensorflow code, then dump model again"
            )
        epsilon = round(node.get_attr('epsilon'), 6)

        if gamma.type == 'const':
            value = gamma.get_attr('value')
            shape = value.tensor_shape
            assert len(shape.dim) == 1
            shape = shape.dim[0].size

            assert len(value.float_val) == 1
            value = value.float_val[0]
            code = "{} = fluid.layers.batch_norm({}, epsilon={}, param_attr=fluid.ParamAttr(\'{}\', fluid.initializer.Constant({})), bias_attr=\'{}\', moving_mean_name=\'{}\', moving_variance_name=\'{}\', is_test=True)".format(
                node.ref_name, inputs.ref_name, epsilon, gamma.ref_name, value,
                beta.ref_name, mv_mean.ref_name, mv_variance.ref_name)
        else:
            code = '{} = fluid.layers.batch_norm({}, epsilon={}, param_attr=\'{}\', bias_attr=\'{}\', moving_mean_name=\'{}\', moving_variance_name=\'{}\', is_test=True)'.format(
                node.ref_name, inputs.ref_name, epsilon, gamma.ref_name,
                beta.ref_name, mv_mean.ref_name, mv_variance.ref_name)
        return code

    def emit_assign(self, node):
        ref = self.graph.get_node(node.layer.input[0])
        value = self.graph.get_node(node.layer.input[1])
        code = 'fluid.layers.assign(input={}, output={})'.format(
            value.ref_name, ref.ref_name)
        return code

    def emit_add(self, node):
        input1 = self.graph.get_node(node.layer.input[0])
        input2 = self.graph.get_node(node.layer.input[1])
        code = '{} = fluid.layers.elementwise_add({}, {})'.format(
            node.ref_name, input1.ref_name, input2.ref_name)
        return code

    def emit_mean(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        reduce_idx = self.graph.get_node(node.layer.input[1])
        idxs = reduce_idx.layer.attr['value'].tensor
        idxs = tensor_util.MakeNdarray(idxs)
        shape = idxs.shape
        if len(shape) != 1:
            raise Exception('Unexpected situation[mean_op]')

        input_shape = self.tensor_shape_to_list(
            inputs.get_attr('_output_shapes'))[0]
        if node.dataformat == "NHWC" and len(input_shape) == 4:
            for i in range(0, shape[0]):
                if idxs[i] == 1:
                    idxs[i] = 2
                elif idxs[i] == 2:
                    idxs[i] = 3
                elif idxs[i] == 3:
                    idxs[i] = 1

        code = '{} = fluid.layers.reduce_mean({}, {}, keep_dim=True)'.format(
            node.ref_name, inputs.ref_name, list(idxs))
        return code

    def emit_squeeze(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        axis = node.get_attr('squeeze_dims')
        input_shape = self.tensor_shape_to_list(
            inputs.get_attr('_output_shapes'))[0]

        if node.dataformat == "NHWC" and len(input_shape) == 4:
            for i in range(0, len(axis)):
                if axis[i] == 1:
                    axis[i] = 2
                elif axis[i] == 2:
                    axis[i] = 3
                elif axis[i] == 3:
                    axis[i] = 1
        code = '{} = fluid.layers.squeeze({}, {})'.format(
            node.ref_name, inputs.ref_name, axis)
        return code

    def emit_const(self, node):
        shape = self.tensor_shape_to_list(node.get_attr('_output_shapes'))[0]
        dtype = node.get_attr('dtype')
        # TODO dtype need more validation
        value = node.layer.attr['value'].tensor.int_val[0]
        if dtype in self.dtype_map:
            dtype = self.dtype_map[dtype]
        else:
            raise Exception('Unknow dtype : {}'.format(dtype))

        if node.dataformat == 'NHWC':
            raise Exception("Const: NHWC format not support yet")

        code = "{} = fluid.layers.fill_constant({}, \'{}\', {})".format(
            node.ref_name, shape, dtype, value)
        return code

    def emit_concatv2(self, node):
        inputs = node.layer.input
        inputs_vars = []
        code = []
        for i in range(0, len(inputs) - 1):
            tmp = self.graph.get_node(inputs[i])
            if tmp.type == 'const':
                code.append(self.emit_const(tmp))
            inputs_vars.append(tmp.ref_name)
        axis = self.graph.get_node(
            inputs[-1]).layer.attr['value'].tensor.int_val[0]

        output_shape = self.tensor_shape_to_list(
            node.get_attr('_output_shapes'))[0]
        if node.dataformat == "NHWC" and len(output_shape) == 4:
            if axis == 1:
                axis = 2
            elif axis == 2:
                axis = 3
            elif axis == 3:
                axis = 1
        code.append('{} = fluid.layers.concat([{}], {})'.format(
            node.ref_name, ', '.join(inputs_vars), axis))
        return code

    def emit_avgpool(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        padding_mode = node.get_attr('padding')
        # TODO need more validation in nlp
        strides = node.get_attr('strides')[1:3]
        pool_size = node.get_attr('ksize')[1:3]
        padding = [0, 0]
        if padding_mode == 'SAME':
            pad_0 = (pool_size[0] - 1) / 2
            pad_1 = (pool_size[1] - 1) / 2
            padding = [pad_0, pad_1]
        code = '{} = fluid.layers.pool2d({}, {}, \'avg\', {}, {})'.format(
            node.ref_name, inputs.ref_name, pool_size, strides, padding)
        return code

    def emit_sub(self, node):
        input1 = self.graph.get_node(node.layer.input[0])
        input2 = self.graph.get_node(node.layer.input[1])
        code = '{} = fluid.layers.elementwise_sub({}, {})'.format(
            node.ref_name, input1.ref_name, input2.ref_name)
        return code

    def emit_mul(self, node):
        input1 = self.graph.get_node(node.layer.input[0])
        input2 = self.graph.get_node(node.layer.input[1])
        code = '{} = fluid.layers.elementwise_mul({}, {})'.format(
            node.ref_name, input1.ref_name, input2.ref_name)
        return code

    def emit_floor(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        code = '{} = fluid.layers.floor({})'.format(node.ref_name,
                                                    inputs.ref_name)
        return code

    def emit_realdiv(self, node):
        input1 = self.graph.get_node(node.layer.input[0])
        input2 = self.graph.get_node(node.layer.input[1])
        code = '{} = fluid.layers.elementwise_div({})'.format(
            node.ref_name, input1.ref_name, input2.ref_name)
        return code

    def emit_shape(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        if "num_split" in inputs.layer.attr:
            code = '{} = fluid.layers.shape({}[0])'.format(
                node.ref_name, inputs.ref_name)
        else:
            code = '{} = fluid.layers.shape({})'.format(
                node.ref_name, inputs.ref_name)
        # TODO there's dtype problem of PaddlePaddle's OP[fluid.layers.shape]
        # https://github.com/PaddlePaddle/Paddle/issues/15267
        # tensorflow2paddle fix problem temporary
        code = [code]
        code.append("{} = fluid.layers.cast({}, dtype='int32')".format(
            node.ref_name, node.ref_name))
        return code

    def emit_stridedslice(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        begin = self.graph.get_node(node.layer.input[1])
        end = self.graph.get_node(node.layer.input[2])
        strides = self.graph.get_node(node.layer.input[3])

        begin = list(tensor_util.MakeNdarray(begin.layer.attr['value'].tensor))
        end = list(tensor_util.MakeNdarray(end.layer.attr['value'].tensor))
        strides = list(
            tensor_util.MakeNdarray(strides.layer.attr['value'].tensor))

        if len(begin) != len(strides) or len(end) != len(strides):
            raise Exception("length of begin/end/strides must be equl")

        for i in range(0, len(strides)):
            if strides[i] != 1:
                raise Exception(
                    "strides must be 1 for all axis, other situation  not supported yet"
                )

        code = "{} = fluid.layers.slice({}, axes={}, starts={}, ends={})".format(
            node.ref_name, inputs.ref_name, [i for i in range(0, len(begin))],
            begin, end)
        return code

    def emit_gather(self, node):
        embedding = self.graph.get_node(node.layer.input[0])
        idxs = self.graph.get_node(node.layer.input[1])

        idxs_shape = self.tensor_shape_to_list(
            idxs.get_attr('_output_shapes'))[0]
        embedding_shape = self.tensor_shape_to_list(
            embedding.get_attr('_output_shapes'))[0]

        if len(embedding_shape) != 2:
            raise Exception("rank of input[0] must be equal to 2 in Gather OP")

        code = []
        if idxs_shape[-1] != 1:
            code.append(
                "reshape = fluid.layers.reshape({}, shape=[-1, 1])".format(
                    idxs.ref_name))
            code.append("gather = fluid.layers.gather({}, reshape)".format(
                embedding.ref_name))
            code.append("{} = fluid.layers.reshape(gather,{})".format(
                node.ref_name, idxs_shape + [embedding_shape[-1]]))
        else:
            code = "{} = fluid.layers.gather({}, {})".format(
                node.ref_name, embedding.ref_name, idxs.ref_name)
        return code

    def emit_transpose(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        perm = self.graph.get_node(node.layer.input[1])
        assert perm.type == "const"
        perm = perm.layer.attr['value'].tensor
        perm = tensor_util.MakeNdarray(perm)

        # TODO
        if node.dataformat == "NHWC" and perm.shape[0] == 4:
            raise Exception(
                "Unsupported situation for op Transpose, NHWC not supported yet"
            )

        perm = list(perm)
        code = "{} = fluid.layers.transpose({}, {})".format(
            node.ref_name, inputs.ref_name, perm)
        return code

    def emit_reshape(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        shape = self.graph.get_node(node.layer.input[1])
        assert shape.type == "const"

        # TODO
        if node.dataformat == "NHWC":
            raise Exception(
                "Unsupported situation for reshape, NHWC not supported yet")

        shape = shape.layer.attr['value'].tensor
        shape = list(tensor_util.MakeNdarray(shape))
        code = "{} = fluid.layers.reshape({}, {})".format(
            node.ref_name, inputs.ref_name, shape)
        return code

    def emit_split(self, node):
        inputs = self.graph.get_node(node.layer.input[1])
        split_dim = self.graph.get_node(node.layer.input[0])
        inputs_shape = self.tensor_shape_to_list(
            inputs.get_attr('_output_shapes'))[0]
        assert split_dim.type == 'const' and len(inputs_shape) > 1
        axis = split_dim.layer.attr['value'].tensor.int_val[0]
        num_split = node.get_attr('num_split')

        code = list()
        if inputs_shape[axis] < 0:
            tmp_shape = [-1, num_split
                         ] + inputs_shape[:axis] + inputs_shape[axis + 1:]
            code.append("reshape = fluid.layers.reshape({}, {})".format(
                inputs.ref_name, tmp_shape))
            code.append(
                "split = fluid.layers.split(reshape, {}, 1)".format(num_split))
            code.append(
                "{} = [fluid.layers.squeeze(s, [1]) for s in split]".format(
                    node.ref_name))
        else:
            code = "{} = fluid.layers.split({}, {}, {})".format(
                inputs.ref_name, num_split, axis)

        return code

    def emit_expanddims(self, node):
        inputs = self.graph.get_node(node.layer.input[0])
        dim = self.graph.get_node(node.layer.input[1])
        dim = tensor_util.MakeNdarray(dim.layer.attr['value'].tensor)
        inputs_shape = self.tensor_shape_to_list(
            inputs.get_attr('_output_shapes'))[0]

        inputs_shape.insert(dim, 1)
        code = "{} = fluid.layers.reshape({}, {})".format(
            node.ref_name, inputs.ref_name, inputs_shape)
        return code

    def emit_fill(self, node):
        value = self.graph.get_node(node.layer.input[1])
        value = value.layer.attr['value'].tensor.float_val[0]
        dtype = node.layer.attr['T'].type
        if dtype in self.dtype_map:
            dtype = self.dtype_map[dtype]
        else:
            raise Exception('Unknow dtype : {}'.format(dtype))

        output_shape = self.tensor_shape_to_list(
            node.get_attr('_output_shapes'))[0]
        code = "{} = fluid.layers.create_parameter({}, {}, default_initializer=fluid.initializer.Constant({}))".format(
            node.ref_name, output_shape, dtype, value)
        return code
