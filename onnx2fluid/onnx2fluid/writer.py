#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 20:44:43 2019

@author: Macrobull
"""

from __future__ import division

import logging, os
import numpy as np

from collections import OrderedDict as Dict

logger = logging.getLogger(__name__)

from . import symbolic

try:
    import paddle.fluid.proto.framework_pb2 as framework_pb2
except ImportError:
    from . import framework_pb2

    logger.warning('importing paddle.fluid.proto.framework_pb2d failed,'
                   'using fallback framework_pb2')

__all__ = [
    'Program',
    'Writer',
]


def irepr(obj, to='_'):
    """inline repr"""

    s = repr(obj)
    for c in '\r\n':
        s = s.replace(c, to)
    if len(s) > 78:
        s = s[:75] + '...'
    return s


def flatten_list(obj, out=None):
    assert isinstance(obj, list), 'list type required'

    if out is None:
        out = type(obj)()
    for item in obj:
        if isinstance(item, list):
            flatten_list(item, out)
        else:
            out.append(item)
    return out


def make_attr_name(name):
    """
    make a valid code name for ParamAttr
    """

    assert name != '', 'name should not be empty'

    for s in ' \\|/:.-':  #
        name = name.replace(s, '_')
    if not name.startswith('_'):
        name = '_' + name
    return 'attr' + name


class Program(object):
    """
    fluid Python code and ProgramDesc wrapper
    """

    DTYPE_TO_FRAMEWORK_DTYPE = {
        'bool': framework_pb2.VarType.BOOL,
        'int8': framework_pb2.VarType.INT8,
        'uint8': framework_pb2.VarType.UINT8,
        'int16': framework_pb2.VarType.INT16,
        'int32': framework_pb2.VarType.INT32,
        'int64': framework_pb2.VarType.INT64,
        'float16': framework_pb2.VarType.FP16,
        'float32': framework_pb2.VarType.FP32,
        'float64': framework_pb2.VarType.FP64
    }

    @staticmethod
    def Dtype(dtype):
        """
        convert dtype to fulid framework dtype
        """

        dtype = np.dtype(dtype).name
        return Program.DTYPE_TO_FRAMEWORK_DTYPE[dtype]

    @staticmethod
    def OpDescVars(keys, vals):
        """
        make (OpDesc.Var)s
        """

        od_vars = []
        for idx, key in enumerate(keys):
            od_var = framework_pb2.OpDesc.Var()
            od_var.parameter = key
            if idx < len(vals):
                od_var.arguments.append(vals[idx])  #
            od_vars.append(od_var)
        return od_vars

    @staticmethod
    def OpDescAttrs(attrs):
        """
        make (OpDesc.Attr)s
        """

        od_attrs = []
        for key, value in attrs.items():
            od_attr = framework_pb2.OpDesc.Attr()
            od_attr.name = key
            if isinstance(value, bool):  # bool.mro() = [bool, int, object]
                od_attr.type = framework_pb2.BOOLEAN
                od_attr.b = value
            elif isinstance(value, int):  # only cast to int32
                od_attr.type = framework_pb2.INT
                od_attr.i = value
            elif isinstance(value, float):
                od_attr.type = framework_pb2.FLOAT
                od_attr.f = value
            elif isinstance(value, str):
                od_attr.type = framework_pb2.STRING
                od_attr.s = value
            elif isinstance(value, list):
                if value:  # TODO: test all items
                    if isinstance(value[0],
                                  bool):  # bool.mro() = [bool, int, object]
                        od_attr.type = framework_pb2.BOOLEANS
                        od_attr.bools.extend(value)
                    elif isinstance(value[0], int):  # only cast to int32 list
                        od_attr.type = framework_pb2.INTS
                        od_attr.ints.extend(value)
                    elif isinstance(value[0], float):
                        od_attr.type = framework_pb2.FLOATS
                        od_attr.floats.extend(value)
                    elif isinstance(value[0], str):
                        od_attr.type = framework_pb2.STRINGS
                        od_attr.strings.extend(value)
                    else:
                        raise ValueError('unsupported attribute {} = {}'.format(
                            key, value))
                else:  # WORKAROUND: [] not inferred
                    #                    raise ValueError('unsupported attribute {} = {}'.format(key, value))
                    od_attr.type = framework_pb2.INTS
                    logger.warning('using attribute %s = %s as INTS', key,
                                   value)
            else:
                raise ValueError('unsupported attribute {} = {}'.format(
                    key, value))
            od_attrs.append(od_attr)
        return od_attrs

    def __init__(self):
        self.code_mutable = True
        self.codes = []
        self.op_descs = []
        self.var_descs = Dict()

    def __repr__(self):
        return ('Program(code mutable: {}) with:\n'
                'codes: {}\n'
                'op_descs: {}\n'
                'var_descs: {}\n').format(self.code_mutable, self.codes,
                                          self.op_descs,
                                          list(self.var_descs.values()))

    def Code(self, code):
        """
        add Python code
        """

        if self.code_mutable:
            self.codes.append(code)

    def OpDesc(self, op_type, input_key_vals, output_key_vals, attrs):
        """
        add OpDesc
        """

        desc = framework_pb2.OpDesc()
        desc.type = op_type
        desc.inputs.extend(self.OpDescVars(*input_key_vals))
        desc.outputs.extend(self.OpDescVars(*output_key_vals))
        desc.attrs.extend(self.OpDescAttrs(attrs))
        self.op_descs.append(desc)
        return desc

    def VarDesc(self,
                name,
                persistable=False,
                value_info=None,
                remove_batch=None):
        """
        add VarDesc,
        """

        assert name not in self.var_descs, 'var name {} conflicts'.format(name)

        var_desc = framework_pb2.VarDesc()
        var_desc.name = name
        var_desc.persistable = persistable
        var_desc.type.type = framework_pb2.VarType.LOD_TENSOR
        self.var_descs[name] = var_desc

        if value_info is not None:
            self.VarTypeShapeInfo(name, value_info, remove_batch=remove_batch)

    def Op(self, domain, op_type, inputs, outputs, attrs, *args, **kwargs):
        """
        convert an ONNX op and add it to program
        """

        if domain != '':  # TODO: symbolic file routing by domain
            raise ValueError('only default domain supported')

        if op_type in symbolic.DEFAULT_OP_MAPPING:
            symbolic._default(self, op_type, inputs, outputs, attrs, *args,
                              **kwargs)
        elif hasattr(symbolic, op_type):
            fn = getattr(symbolic, op_type)
            fn(self, inputs, outputs, attrs, *args, **kwargs)
        else:
            raise ValueError('conversion for {}::{} not supported'.format(
                domain, op_type))

    def IntermediateOp(self, domain, op_type, inputs, outputs, attrs, *args,
                       **kwargs):
        """
        convert an intermediate ONNX op declaring in desc program only
        """

        code_mutable = self.code_mutable
        self.code_mutable = False
        try:
            self.Op(domain, op_type, inputs, outputs, attrs, *args, **kwargs)
        except BaseException as e:
            self.code_mutable = code_mutable
            raise e
        else:
            self.code_mutable = code_mutable

    def VarTypeShapeInfo(self, name, value_info, remove_batch=None):
        """
        set value_info for var
        """

        if name not in self.var_descs:
            return

        dtype = value_info.get('dtype', None)
        if dtype is None:
            return

        var_desc = self.var_descs[name]
        tensor_desc = var_desc.type.lod_tensor.tensor
        tensor_desc.data_type = self.Dtype(dtype)  # required

        shape = value_info.get('shape', None)
        if not shape:  # None or scalars
            return

        tensor_desc.dims.extend(shape)
        if remove_batch is None:
            remove_batch = value_info.get('remove_batch',
                                          False)  #not persistable)
        if remove_batch:
            tensor_desc.dims[0] = -1


class Writer(object):
    """
    fluid code and desc writter
    """

    CODE_INDENT = ' ' * 4  # '\t'

    @staticmethod
    def header_code(func_name, info=''):
        """
        Python header codes
        """

        codes = []
        codes.append('"""')
        codes.append('This code is generated by onnx2fluid.')
        codes.append('{}'.format(info))
        codes.append('"""')
        codes.append('')
        codes.append('from __future__ import division')
        codes.append('')
        codes.append('from paddle.fluid import ParamAttr')
        codes.append('from paddle.fluid import initializer, layers')
        codes.append('')
        codes.append('')

        codes.append('def {}():'.format(func_name))
        return codes

    @staticmethod
    def emit_op(prog, name, domain, op_type, inputs, outputs, attrs,
                value_infos, *args, **kwargs):
        """
        emit an ONNX op into program
        """

        prog.Code('# {}, {}::{}: {} -> {}, {}'.format(name, domain, op_type,
                                                      inputs, outputs,
                                                      irepr(attrs, to=', ')))
        prog.Op(domain,
                op_type,
                inputs,
                outputs,
                attrs,
                value_infos=value_infos,
                name=name,
                *args,
                **kwargs)

    @staticmethod
    def emit_param(prog, name, value_info):
        """
        emit an ONNX weight into program
        """

        embedded_names = value_info.get('embedded_as', [])
        if embedded_names:
            prog.Code('# parameter {} embedded as {}'.format(
                name, embedded_names))
            for embedded_name in embedded_names:
                prog.VarDesc(embedded_name,
                             persistable=True,
                             value_info=value_info)
        else:
            attr_name = make_attr_name(name)
            prog.Code('# parameter {}'.format(name))
            prog.Code('{} = ParamAttr(name={})'  # , trainable=True
                      .format(attr_name, repr(name)))
            prog.Code(
                '{} = layers.create_parameter(shape={}, dtype={}, name={}, attr={}'
                ', default_initializer=initializer.Constant(0))'  #, is_bias={}
                .format(name, value_info['shape'],
                        repr(value_info['dtype'].name), repr(name),
                        attr_name))  #, value_info.get('is_bias', False)))
            prog.VarDesc(name, persistable=True, value_info=value_info)

    @staticmethod
    def emit_inputs(prog, names, value_infos, remove_batch=None):
        """
        emit ONNX inputs into program
        """

        for idx, name in enumerate(names):
            value_info = value_infos[name]
            shape = value_info['shape']
            if remove_batch is None:
                remove_batch = value_info.get('remove_batch',
                                              True)  # HINT: True by default ?
            if remove_batch:
                shape = shape[1:]

            prog.Code('# input {}'.format(name))
            prog.Code((
                '{} = layers.data(name={}, shape={}, dtype={}, '
                'append_batch_size={})'  # , stop_gradient=True
            ).format(
                name,
                repr(name),
                shape,
                repr(value_info['dtype'].name),
                remove_batch,
            ))
            prog.OpDesc(
                'feed',
                (['X'], ['feed']),
                (['Out'], [name]),
                {'col': idx},
            )
            prog.VarDesc(name, value_info=value_info, remove_batch=remove_batch)

    @staticmethod
    def emit_outputs(prog, names):  #, value_infos
        """
        emit ONNX outputs into program
        """

        code = 'return '
        for idx, name in enumerate(names):
            code += name + ', '

            prog.OpDesc(
                'fetch',
                (['X'], [name]),
                (['Out'], ['fetch']),
                {'col': idx},
            )
            # var is emitted over ops
        prog.Code(code)

    @staticmethod
    def add_codes(codes, others, indent):
        """
        flatten codes in program
        """

        for code in flatten_list(others):
            codes.append(Writer.CODE_INDENT * indent + code)
        return codes

    @staticmethod
    def write_weight(weight, filename, lod=None):
        """
        write single weight in fluid desc
        """

        assert isinstance(weight, np.ndarray), 'weight is not an ndarray'
        assert lod is None or isinstance(lod,
                                         list), 'lod should be None or list'

        if lod is None:
            lod = [0]

        tensor_desc = framework_pb2.VarType.TensorDesc()
        tensor_desc.data_type = Program.Dtype(weight.dtype)
        tensor_desc.dims.extend(weight.shape)

        fp = open(filename, 'wb')
        np.array([0], dtype=np.int32).tofile(fp)  # version
        np.array(lod, dtype=np.int64).tofile(fp)  # LOD level
        np.array([0], dtype=np.int32).tofile(fp)  # tensor version
        np.array([tensor_desc.ByteSize()], dtype=np.int32).tofile(fp)
        fp.write(tensor_desc.SerializeToString())
        weight.tofile(fp)
        fp.close()

    @staticmethod
    def write_weights(weights, save_dir):
        """
        write multiple weights in each fluid desc
        """

        for name, weight in weights.items():
            assert isinstance(weights, dict), 'dict type weights required'

            filename = os.path.join(save_dir, name)
            Writer.write_weight(weight, filename)
            logger.debug('saved weight %s to %s', name, filename)

    @staticmethod
    def write_code_file(filename, header_code, *body_codes):
        """
        write Python code to file
        """

        codes = []
        Writer.add_codes(codes, header_code, 0)
        for body_code in body_codes:
            Writer.add_codes(codes, body_code, 1)

        fp = open(filename, 'w')
        for code in flatten_list(codes):
            fp.write(code)
            fp.write('\n')
        fp.close()
        logger.debug('saved codes to %s', filename)

    @staticmethod
    def write_desc_file(filename, op_descs, var_descs):
        """
        write desc program to file
        """

        prog_desc = framework_pb2.ProgramDesc()
        block_desc = prog_desc.blocks.add()
        block_desc.idx = 0
        block_desc.parent_idx = -1
        block_desc.ops.extend(op_descs)
        block_desc.vars.extend(var_descs)

        # add feed-fetch on vars
        feed_var_desc = block_desc.vars.add()
        feed_var_desc.name = 'feed'
        feed_var_desc.type.type = framework_pb2.VarType.FEED_MINIBATCH
        feed_var_desc.persistable = True
        fetch_var_desc = block_desc.vars.add()
        fetch_var_desc.name = 'fetch'
        fetch_var_desc.type.type = framework_pb2.VarType.FETCH_LIST
        fetch_var_desc.persistable = True

        fp = open(filename, 'wb')
        fp.write(prog_desc.SerializeToString())
        fp.close()
        logger.debug('saved descs to %s', filename)
