#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX to Paddle fluid symbolic translation

TODO: move non-ONNX ops out to symbolic_aten.py, symbolic_caffe2.py ...

Created on Mon Feb 25 09:33:43 2019

@author: Macrobull
"""

from __future__ import division

import logging as _logging
import numpy as _np

from collections import OrderedDict as _dict
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

_logger = _logging.getLogger(__name__)

ONNX_INT_MAX = 2**63 - 1
FLUID_INT_MAX = 2**31 - 1  #

DEFAULT_ONNX_OP_DOMAIN = ''
DEFAULT_FLUID_OP_NAMESCOPE = '/'

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
DEFAULT_OP_MAPPING_VALUES = list(DEFAULT_OP_MAPPING_FIELD_VALUES.values())

DEFAULT_OP_MAPPING = {
        ## nil ops ##
        'RandomUniform':
            ['uniform_random', [], ['Out'], dict(high='max', low='min'),
             dict(), None, None, False], # TODO: add dtype support
        'RandomNormal':
            ['gaussian_random', [], ['Out'], dict(scale='std'),
             dict(), None, None, False], # TODO: add dtype support
        ## unary ops ##
        'Abs': ['abs', ['X'], ['Out']],
        'Acos': ['acos', ['X'], ['Out']],
        'Asin': ['asin', ['X'], ['Out']],
        'Atan': ['atan', ['X'], ['Out']],
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
        'HardSigmoid': ['hard_sigmoid', ['X'], ['Out'], dict(alpha='slope', beta='offset')],
        'Identity': ['assign', ['X'], ['Out']],
        'LeakyRelu': ['leaky_relu', ['X'], ['Out']],
        'Log': ['log', ['X'], ['Out']],
        'LRN': ['lrn', ['X'], ['Out', 'MidOut'], dict(size='n', bias='k')], #
        'Reciprocal': ['reciprocal', ['X'], ['Out']],
        'Relu': ['relu', ['X'], ['Out']],
        'Round': ['round', ['X'], ['Out']],
        'Selu': ['selu', ['X'], ['Out'], dict(gamma='scale')],
        'Shape': ['shape', ['X'], ['Out']], # FIXME: out is int64 vs int32
        'Shrink': ['softshrink', ['X'], ['Out'], dict(bias='', labmd='')],
        'Sigmoid': ['sigmoid', ['X'], ['Out']],
        'Sign': ['sign', ['X'], ['Out']],
        'Sin': ['sin', ['X'], ['Out']],
        'Squeeze': ['squeeze', ['X'], ['Out']], # attrs bypassed, FIXME: emit squeeze2
        'Softplus': ['softplus', ['X'], ['Out']],
        # FIXME: default axis = -1, reshape required before and after
        'Softmax': ['softmax', ['X'], ['Out'], dict(axis='')],
        'Softplus': ['softplus', ['X'], ['Out']],
        'Softsign': ['softsign', ['X'], ['Out']],
        'SpaceToDepth': ['space_to_depth', ['X'], ['Out']],
        'Sqrt': ['sqrt', ['X'], ['Out']],
        'Tanh': ['tanh', ['X'], ['Out']],
        'ThresholdedRelu': ['thresholded_relu', ['X'], ['Out'], dict(alpha='threshold')],
        #'Transpose': ['transpose', ['X'], ['Out']],
        'Unsqueeze': ['unsqueeze', ['X'], ['Out']], # attrs bypassed, FIXME: emit unsqueeze2
        ## binary ops ##
        'Add': ['elementwise_add', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        #'AffineGrid': ['affine_grid', ['Theta'], ['Output'], dict(size='out_shape')],
        'And': ['logical_and', ['X', 'Y'], ['Out']],
        'Div': ['elementwise_div', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        'Equal': ['equal', ['X', 'Y'], ['Out'], dict(), dict(), None, None, False],
        'Greater': ['less_than', ['X', 'Y'], ['Out'], dict(), dict(), [1, 0], None, False],
        'Less': ['less_than', ['X', 'Y'], ['Out'], dict(), dict(), None, None, False],
        'MatMul': ['matmul', ['X', 'Y'], ['Out']], # defaults excluded for transpose_x vs transpose_X
        'Max': ['elementwise_max', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        'Min': ['elementwise_min', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        'Mod': ['elementwise_mod', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        'Mul': ['elementwise_mul', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
        'Not': ['logical_not', ['X', 'Y'], ['Out']],
        'OneHot': # assuming values=[0, 1], axis=-1 and drop them
            ['one_hot', ['Input', 'Depth'], ['Out'], dict(axis=''), dict(),
             [0, 1], None, False],
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
}

DEFAULT_IOA_CONSTRAINTS = {
    'ArgMax': [
        (lambda i, o, a: a.get('keepdims', 1) == 1,
         'only keepdims = 0 supported'),
    ],
    'ArgMin': [
        (lambda i, o, a: a.get('keepdims', 1) == 1,
         'only keepdims = 0 supported'),
    ],
    'Gather': [
        (lambda i, o, a: a.get('axis', 0) == 0, 'only axis = 0 supported'),
    ],
    'Shrink': [
        (lambda i, o, a: a.get('bias', 0) == a.get('lambd', 0.5),
         'only SoftShrink with bias = lambd supported'),
    ],
    #        'Softmax':
    #            [(lambda i, o, a: a.get('axis', 1) == -2, 'Paddle fluid Softmax works on dim -2 only'),
    #            ],
    'OneHot': [
        (lambda i, o, a: a.get('axis', -1) == -1, 'only axis = -1 supported'),
    ],
    'Scatter': [
        (lambda i, o, a: a.get('axis', 0) == 0, 'only axis = 0 supported'),
    ],
    'TopK': [
        (lambda i, o, a: a.get('axis', -1) == -1, 'only axis = -1 supported'),
    ],
}


def _dtype(value_infos, name):
    return _np.dtype(value_infos[name]['dtype'])


def _dtype_or_none(value_infos, name):
    if name not in value_infos:
        return None
    value_info = value_infos[name]
    if 'dtype' not in value_info:
        return None
    return _np.dtype(value_info['dtype'])


def _shape(value_infos, name):
    return list(value_infos[name]['shape'])


def _shape_or_none(value_infos, name):
    if name not in value_infos:
        return None
    value_info = value_infos[name]
    if 'shape' not in value_info:
        return None
    return list(value_info['shape'])


def _const_weight_or_none(value_infos, name):
    if name not in value_infos:
        return None
    value_info = value_infos[name]
    const_value = value_info.get('const_value', None)
    if const_value is not None:
        return const_value
    get_weight_func = value_info.get('get_weight', None)
    if get_weight_func is not None:
        return get_weight_func()
    return None


def _check_embeddable(value_infos, *names):
    keyword = 'get_weight'
    for name in names:
        if keyword not in value_infos[name]:
            _logger.warning('parameter %s not embeddable', name)
            return False
    return True


def _default(prog, op_type, inputs, outputs, attrs, *args, name='', **kwargs):
    info = DEFAULT_OP_MAPPING[op_type]
    info.extend(DEFAULT_OP_MAPPING_VALUES[len(info):])

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

    if fluid_op in DEFAULT_IOA_CONSTRAINTS:
        for predicate, message in DEFAULT_IOA_CONSTRAINTS[fluid_op]:
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

    var_inps = list(map(inputs.__getitem__,
                        input_perm)) if input_perm else inputs
    var_outs = list(map(outputs.__getitem__,
                        output_perm)) if output_perm else outputs
    arg_name = ', name={}'.format(
        repr(name)) if fill_name_field and name else ''
    arg_attrs = [
        ', {}={}'.format(key, value) for key, value in fluid_attrs.items()
    ]

    prog.Code('{} = layers.{}({}{}{})'.format(
        ', '.join(var_outs),
        fluid_op,
        ', '.join(var_inps),
        ''.join(arg_attrs)[(0 if var_inps else 2):],
        arg_name,
    ))

    # dummy var_out
    num_vars = len(var_outs)
    num_args = len(fluid_output_args)
    if num_vars < num_args:
        assert fill_name_field, 'name required to name dummy output variables'
        for idx_out in range(num_vars, num_args):
            var_out = name + '.' + fluid_output_args[idx_out]  # dummy output
            var_outs.append(var_out)

    for var_out in var_outs:
        prog.VarDesc(var_out)
    prog.OpDesc(fluid_op, (fluid_input_args, var_inps),
                (fluid_output_args, var_outs), fluid_attrs)


def _assign(prog, mapping):
    fluid_op = 'assign'

    for var_dst, var_src in mapping.items():
        prog.Code('{} = {} # assign'.format(var_dst, var_src))
        #        prog.Code('{} = layers.{}({})'
        #                  .format(var_dst,
        #                          fluid_op,
        #                          var_src,
        #                          ))
        prog.VarDesc(var_dst)
        prog.OpDesc(
            fluid_op,
            (['X'], [var_src]),
            (['Out'], [var_dst]),
            dict(),
        )


def _zeros_like(prog, var_ref, var_out, value_infos):
    prog.Op(
        '',
        'Sub',
        [var_ref, var_ref],
        [var_out],
        {'axis': 0},
        value_infos,
    )


def _pad_if_asymmetric(prog, pads, var_input, value_infos):  # pads: SSEE
    assert len(pads) & 1 == 0
    ndims = len(pads) // 2
    symmetric = True
    for idx_dim in range(ndims):
        if pads[idx_dim] != pads[ndims + idx_dim]:
            symmetric = False
            break
    if symmetric:
        return pads[:ndims], var_input

    var_padded = var_input + '_padded'  # explicit variable
    prog.Op(
        '',
        'Pad',
        [var_input],
        [var_padded],
        {
            'mode': 'constant',
            'value': 0.,
            'pads': pads,
        },
        value_infos=value_infos,
        name=(var_input + '_pad'),
    )
    return [0] * ndims, var_padded


def _adaptive_pool(prog, pool_type, inputs, outputs, attrs, name=''):
    # I/O
    var_x, = inputs
    var_y, var_indices, = (outputs + [None] * 1)[:2]

    # interpretation
    pool_size = attrs['output_size']  # required
    poolnd = len(pool_size)
    assert 2 <= poolnd <= 3, 'only pool2d and pool3d supported'

    fluid_op = 'adaptive_pool{}d'.format(poolnd)
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('{}{} = layers.{}({}'
              ', require_index={}'
              ', pool_size={}'
              ', pool_type={}'
              '{})'.format(
                  var_y,
                  ', {}'.format(var_indices) if var_indices else '',
                  fluid_op,
                  var_x,
                  # attrs
                  bool(var_indices),
                  pool_size,
                  repr(pool_type),
                  name_attr,
              ))
    fluid_op = 'pool{}d'.format(poolnd)
    prog.VarDesc(var_y)
    if var_indices:
        prog.VarDesc(var_indices)
    prog.OpDesc(
        fluid_op,
        (['X'], [var_x]),
        (['Out', 'Indices'], [var_y] + ([var_indices] if var_indices else [])),
        {
            'global_pooling': False,
            'adaptive': True,
            'require_index': bool(var_indices),
            'pooling_type': pool_type,
            'ksize': pool_size,
            # unused
            #                    'exclusive': True,
        },
    )


def _global_pool(prog, pool_type, inputs, outputs, attrs, value_infos, name=''):
    # I/O
    var_x, = inputs
    var_y, = outputs

    # interpretation
    input_shape = _shape_or_none(value_infos, var_x)
    output_shape = _shape_or_none(value_infos, var_y)
    assert input_shape is not None or output_shape is not None, 'poolnd not inferred'  # NC...
    if input_shape is not None:
        poolnd = len(input_shape) - 2  # NC...
    elif output_shape is not None:
        poolnd = len(output_shape) - 2  # NC...
    assert 2 <= poolnd <= 3, 'only pool2d and pool3d supported'

    fluid_op = 'pool{}d'.format(poolnd)
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('{} = layers.{}({}, global_pooling=True'
              ', pool_type={}'
              '{})'.format(
                  var_y,
                  fluid_op,
                  var_x,
                  # attrs
                  repr(pool_type),
                  name_attr,
              ))
    prog.VarDesc(var_y)
    prog.OpDesc(
        fluid_op,
        (['X'], [var_x]),
        (['Out'], [var_y]),
        {
            'global_pooling': True,
            'adaptive': False,
            'pooling_type': pool_type,
            'ksize': [-1, -1],
            # unused
            'strides': [-1, -1],
            'paddings': [0, 0],
            'ceil_mode': False,
        },
    )


def _pool(prog, pool_type, inputs, outputs, attrs, value_infos, name=''):
    # I/O
    var_x, = inputs
    var_y, var_indices, = (outputs + [None] * 1)[:2]

    # interpretation
    assert attrs.get(
        'auto_pad',
        'NOTSET') == 'NOTSET', 'only auto_pad = NOTSET supported'  # optional
    assert attrs.get('count_include_pad',
                     0) == 0, 'only count_include_pad = 0 supported'  # optional
    pool_size = attrs['kernel_shape']  # required
    poolnd = len(pool_size)
    assert 2 <= poolnd <= 3, 'only pool2d and pool3d supported'

    fluid_op = 'pool{}d'.format(poolnd)
    strides = attrs.get('strides', [1] * poolnd)  # optional
    ceil_mode = bool(attrs.get('ceil_mode', 0))  # optional
    pads = attrs.get('pads', [0] * (poolnd * 2))  # optional
    paddings, var_x = _pad_if_asymmetric(prog, pads, var_x, value_infos)
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('{} = layers.{}({}, exclusive=True'
              ', pool_size={}'
              ', pool_type={}'
              ', pool_stride={}'
              ', pool_padding={}'
              ', ceil_mode={}'
              '{})'.format(
                  var_y,
                  fluid_op,
                  var_x,
                  # attrs
                  pool_size,
                  repr(pool_type),
                  strides,
                  paddings,
                  ceil_mode,
                  name_attr,
              ))
    prog.VarDesc(var_y)
    if var_indices:
        prog.VarDesc(var_indices)
    prog.OpDesc(
        fluid_op,
        (['X'], [var_x]),
        (['Out', 'Indices'], [var_y] + ([var_indices] if var_indices else [])),
        {
            'global_pooling': False,
            'adaptive': False,
            'require_index': bool(var_indices),
            'pooling_type': pool_type,
            'ksize': pool_size,
            'strides': strides,
            'paddings': paddings,
            'ceil_mode': ceil_mode,
            # unused
            'exclusive': True,
        },
    )


def _roi_pool(prog, fluid_op, inputs, outputs, attrs, value_infos, name):
    # I/O
    var_x, var_rois, = inputs
    var_y, = outputs

    # interpretation
    spatial_scale = attrs['spatial_scale']  # required
    pooled_height, pooled_width = attrs['pooled_shape']  # required
    od_attrs = {
        'pooled_height': pooled_height,
        'pooled_width': pooled_width,
        'spatial_scale': spatial_scale,
    }
    feature_attr = ''
    is_max_pool = fluid_op == 'roi_pool'
    if 'sampling_ratio' in attrs:  #
        sampling_ratio = attrs['sampling_ratio']
        od_attrs['sampling_ratio'] = sampling_ratio
        feature_attr += ', sampling_ratio={}'.format(sampling_ratio)
    if 'output_channels' in attrs:  #
        output_channels = attrs['output_channels']
        od_attrs['output_channels'] = output_channels
        feature_attr += ', output_channels={}'.format(output_channels)

    # generation
    prog.Code('{} = layers.{}({} {}'
              ', spatial_scale={}'
              ', pooled_height={}'
              ', pooled_width={}'
              '{})'.format(
                  var_y,
                  fluid_op,
                  var_x,
                  var_rois,
                  # attrs
                  spatial_scale,
                  pooled_height,
                  pooled_width,
                  feature_attr,
              ))
    prog.VarDesc(var_y)
    if is_max_pool:
        var_argmax = name + '.argmax'  # hidden variable
        prog.VarDesc(var_argmax)
    prog.OpDesc(
        fluid_op,
        (['X', 'Rois'], [var_x, var_rois]),
        (['Out', 'Argmax'], [var_y] + ([var_argmax] if is_max_pool else [])),
        od_attrs,
    )


def _interpolate(prog, inputs, outputs, attrs, value_infos, name=''):
    # I/O
    var_x, var_scales, = inputs
    var_y, = outputs

    # interpretation
    # output shape
    out_shape_ = _shape_or_none(value_infos, var_y)
    if out_shape_ is not None:
        assert len(out_shape_) == 4, 'only 4-D Tensor as X and Y supported'
        out_shape_ = out_shape_[2:]
    # try scales
    scales = _const_weight_or_none(value_infos, var_scales)
    if scales is not None:
        assert len(scales) == 4, 'only 4-D Tensor as X and Y supported'
        assert scales[0] == 1 and scales[
            1] == 1, 'only scale on (NC)HW supported'
        assert scales[2] == scales[
            3], 'only aspect-ratio-invariant scale supported'
    scale = scales and scales[2]
    # try input shape
    if scale is None:
        assert out_shape_, 'neither scales nor output shape available'
        out_shape = out_shape_
    else:
        out_shape = None
        if out_shape_ is None:
            in_shape = _shape_or_none(value_infos, var_x)
            assert in_shape is not None, 'out_shape required but not inferrable'
            assert len(in_shape) == 4, 'only 4-D Tensor as X and Y supported'
            out_shape_ = [in_shape[2] * scale, in_shape[3] * scale]
    mode = attrs.get('mode', 'nearest')
    fluid_op = 'resize_{}'.format(mode)  # not sure bilinear will be linear?
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('{} = layers.{}({}'
              ', scale={}'
              ', out_shape={}'
              '{})'.format(
                  var_y,
                  fluid_op,
                  var_x,
                  # attrs
                  scale,
                  out_shape,
                  name_attr,
              ))
    fluid_op = '{}_interp'.format(mode)
    prog.VarDesc(var_y)
    prog.OpDesc(
        fluid_op,
        (['X'], [var_x]),
        (['Out'], [var_y]),
        {
            'interp_method': mode,
            'out_h ': out_shape_[0],
            'out_w ': out_shape_[1],
        },
    )


def AdaptiveAveragePool(prog, inputs, outputs, attrs, *args, name='', **kwargs):
    """
    aten::adaptive_avg_poolnd
    """

    return _adaptive_pool(prog, 'avg', inputs, outputs, attrs, name=name)


def AdaptiveMaxPool(prog, inputs, outputs, attrs, *args, name='', **kwargs):
    """
    aten::adaptive_max_poolnd
    """

    return _adaptive_pool(prog, 'max', inputs, outputs, attrs, name=name)


def AffineGrid(prog, inputs, outputs, attrs, *args, name='', **kwargs):
    """
    aten::affine_grid
    """

    # I/O
    var_theta, = inputs
    var_grid, = outputs

    # interpretation
    fluid_op = 'affine_grid'
    size = attrs['size']  # required
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('{} = layers.{}({}'
              ', out_shape={}'
              '{})'.format(
                  var_grid,
                  fluid_op,
                  var_theta,
                  # attrs
                  size,
                  name_attr,
              ))
    prog.VarDesc(var_grid)
    prog.OpDesc(
        fluid_op,
        (['Theta'], [var_theta]),
        (['Output'], [var_grid]),
        {'output_shape': size},  # f**k you API
    )


def AveragePool(prog,
                inputs,
                outputs,
                attrs,
                value_infos,
                name='',
                *args,
                **kwargs):
    """
    onnx::AveragePool-10:
    """

    return _pool(prog, 'avg', inputs, outputs, attrs, value_infos, name=name)


def BatchNormalization(prog,
                       inputs,
                       outputs,
                       attrs,
                       value_infos,
                       name='',
                       embed_params=False,
                       *args,
                       **kwargs):
    """
    onnx::BatchNormalization-9:
    """

    # I/O
    var_x, var_scale, var_b, var_mean, var_var, = inputs
    var_y, var_mean_, var_var_, var_saved_mean, var_saved_variance, = (
        outputs + [None] * 4)[:5]
    assert var_saved_mean or (name != '')
    assert var_saved_variance or (name != '')
    var_saved_mean = var_saved_mean or (name + '.saved_mean')  # dummy output
    var_saved_variance = var_saved_variance or (name + '.saved_variance'
                                                )  # dummy output

    # interpretation
    fluid_op = 'batch_norm'
    momentum = attrs.get('momentum', .9)  # optional
    epsilon = attrs.get('epsilon', 1e-5)  # optional
    name_attr = ', name={}'.format(repr(name)) if name else ''
    if embed_params:
        embed_params = _check_embeddable(value_infos, var_scale, var_b,
                                         var_mean, var_var)
        if not embed_params and name:
            _logger.warning('for op  %s(%s -> BatchNormalization -> %s)', name,
                            inputs, outputs)
            _logger.warning('broken Python code will be generated')
    if embed_params:
        assert name != ''
        embedded_scale = name + '.w_0'
        embedded_b = name + '.b_0'
        embedded_mean = name + '.w_1'
        embedded_var = name + '.w_2'
        value_infos[var_scale]['embedded_as'].append(embedded_scale)
        value_infos[var_b]['embedded_as'].append(embedded_b)
        value_infos[var_mean]['embedded_as'].append(embedded_mean)
        value_infos[var_var]['embedded_as'].append(embedded_var)
        var_scale = embedded_scale
        var_b = embedded_b
        var_mean = embedded_mean
        var_var = embedded_var
        param_attr = ''
    else:
        param_attr = (', param_attr={}, bias_attr={}'
                      ', moving_mean_name={}, moving_variance_name={}').format(
                          repr(var_scale), repr(var_b), repr(var_mean),
                          repr(var_var))

    # generation
    prog.Code('{} = layers.{}({}, is_test=True'
              ', momentum={}'
              ', epsilon={}'
              '{}{})'.format(
                  var_y,
                  fluid_op,
                  var_x,
                  # attrs
                  momentum,
                  epsilon,
                  param_attr,
                  name_attr,
              ))
    prog.VarDesc(var_y)
    prog.VarDesc(var_saved_mean)
    prog.VarDesc(var_saved_variance)
    prog.OpDesc(
        fluid_op,
        (['X', 'Scale', 'Bias', 'Mean', 'Variance'
          ], [var_x, var_scale, var_b, var_mean, var_var]),
        (['Y', 'MeanOut', 'SavedMean', 'SavedVariance', 'VarianceOut'
          ], [var_y, var_mean, var_saved_mean, var_saved_variance, var_var]),
        {
            'momentum': momentum,
            'epsilon': epsilon,
            'is_test': 1,
            # unused
        },
    )


def Cast(prog, inputs, outputs, attrs, value_infos, *args, **kwargs):
    """
    onnx::Cast-9:
    """

    # I/O
    var_input, = inputs
    var_output, = outputs

    # interpretation
    dtype = attrs['to']  # required
    if not isinstance(dtype, _np.dtype):  # additional: possible np.dtype
        dtype = TENSOR_TYPE_TO_NP_TYPE[dtype]


#    output_dtype = _dtype_or_none(value_infos, var_output)
#    if output_dtype is not None:
#        assert dtype == output_dtype, 'dtype of to unmatches output'

    fluid_op = 'cast'

    # generation
    prog.Code('{} = layers.{}({}'
              ', dtype={}'
              ')'.format(
                  var_output,
                  fluid_op,
                  var_input,
                  # attrs
                  repr(dtype.name),
              ))
    prog.VarDesc(var_output)
    prog.OpDesc(
        fluid_op,
        (['X'], [var_input]),
        (['Out'], [var_output]),
        {
            'in_dtype': prog.Dtype(_dtype(value_infos,
                                          var_input)),  # holy, required
            'out_dtype': prog.Dtype(dtype),
        },
    )


def Concat(prog, inputs, outputs, attrs, *args, name='', **kwargs):
    """
    onnx::Concat-4:
    """

    # I/O
    var_ret, = outputs

    # interpretation
    fluid_op = 'concat'
    axis = attrs['axis']  # required
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('{} = layers.{}({}'
              ', axis={}'
              '{})'.format(
                  var_ret,
                  fluid_op,
                  '[' + ', '.join(inputs) + ']',
                  # attrs
                  axis,
                  name_attr,
              ))
    prog.VarDesc(var_ret)
    prog.OpDesc(
        fluid_op,
        (['X'] * len(inputs), inputs),
        (['Out'], [var_ret]),
        {'axis': axis},
    )


def Constant(prog, inputs, outputs, attrs, value_infos, *args, **kwargs):
    """
    onnx::Constant-9:
    """

    # I/O
    assert len(inputs) == 0, 'constant op accept no inputs'
    var_output, = outputs

    # interpretation
    value = attrs['value']  # required
    dtype = _np.dtype(value.dtype)
    #    output_dtype = _dtype_or_none(value_infos, var_output)
    #    if output_dtype is not None:
    #        assert dtype == output_dtype, 'tensor dtype unmatches storage dtype'
    #    dtype = _np.dtype('float32') # HINT: force to float32
    shape = attrs.get('shape', None)  # additional
    if shape is None:
        shape = _shape_or_none(value_infos, var_output)
    if shape is None:
        shape = list(value.shape)
        _logger.warning(
            'in op (Constant -> %s): '
            'attribute "shape" of %s not inferred, '
            'using value as 1-D tensor may lead to fails', outputs, var_output)

    # generation
    if len(shape) == 0 or value.size == 1:  # scalar or 1-size
        shape = [1]  # WORKAROUND: bad scalar support
        value = value.tolist()[0]
        fluid_op = 'fill_constant'
        prog.Code('{} = layers.{}(shape={}, dtype={}, value={})'.format(
            var_output,
            fluid_op,
            # attrs
            shape,
            repr(dtype.name),
            value,
        ))
        prog.VarDesc(var_output)
        prog.OpDesc(
            fluid_op,
            ([], []),
            (['Out'], [var_output]),
            {
                'shape': shape,
                'dtype': prog.Dtype(dtype),
                'value': value,
            },
        )
    else:  # list parameter -> const_value
        prog.Code('# {} = {} # passed directly as literal'.format(
            var_output, value.tolist()))

    value_infos[var_output]['const_value'] = value


def ConstantOfShape(prog, inputs, outputs, attrs, value_infos, *args, **kwargs):
    """
    onnx::ConstantOfShape-9:
    """

    # I/O
    var_shape, = inputs
    var_output, = outputs

    shape = _const_weight_or_none(value_infos, var_shape)
    if shape is None:
        shape = _shape_or_none(value_infos, var_output)
    assert shape is not None, (
        'given shape is neither const value nor deductible from output, '
        'this is not supported')
    attrs = attrs.copy()
    attrs.setdefault('value', np.array(0, dtype=np.float32))
    attrs.update({'shape': shape})  # pass const

    prog.Code('# shape:{}={} # const as literal'.format(var_shape, shape))
    prog.Op(
        '',
        'Constant',
        [],
        outputs,
        attrs,
        value_infos,
    )


def Conv(prog,
         inputs,
         outputs,
         attrs,
         value_infos,
         name,
         embed_params=False,
         *args,
         **kwargs):
    """
    onnx::Conv-1:
    """

    # I/O
    var_x, var_w, var_b, = (inputs + [None] * 1)[:3]
    var_y, = outputs

    # interpretation
    assert attrs.get(
        'auto_pad',
        'NOTSET') == 'NOTSET', 'only auto_pad = NOTSET supported'  # optional
    kernel_shape = attrs.get('kernel_shape',
                             _shape(value_infos, var_w)[2:])  # optional, HW
    assert kernel_shape, 'kernel_shape not inferred'
    convnd = len(kernel_shape)
    assert 2 <= convnd <= 3, 'only conv2d and conv3d supported'
    num_out_channels = _shape(value_infos, var_w)[0]  # OI...

    fluid_op = 'conv{}d'.format(convnd)
    num_groups = attrs.get('group', 1)  # optional
    strides = attrs.get('strides', [1] * convnd)  # optional
    dilations = attrs.get('dilations', [1] * convnd)  # optional
    pads = attrs.get('pads', [0] * (convnd * 2))  # optional
    paddings, var_x = _pad_if_asymmetric(prog, pads, var_x, value_infos)
    name_attr = ', name={}'.format(repr(name))
    if embed_params:
        embed_params = _check_embeddable(
            value_infos, *([var_w] + ([var_b] if var_b else [])))
        if not embed_params:
            _logger.warning('for op  %s(%s -> Conv -> %s)', name, inputs,
                            outputs)
            _logger.warning('broken Python code will be generated')
    if embed_params:
        embedded_w = name + '.w_0'
        value_infos[var_w]['embedded_as'].append(embedded_w)
        var_w = embedded_w
        if var_b:
            embedded_b = name + '.b_0'
            value_infos[var_b]['embedded_as'].append(embedded_b)
            var_b = embedded_b
            param_attr = ''
        else:
            param_attr = ', bias_attr=False'
    else:
        param_attr = ', param_attr={}, bias_attr={}'.format(
            repr(var_w),
            repr(var_b) if var_b else False)

    # generation
    prog.Code('{} = layers.{}({}'
              ', num_filters={}'
              ', filter_size={}'
              ', stride={}'
              ', padding={}'
              ', dilation={}'
              ', groups={}'
              '{}{})'.format(
                  var_y,
                  fluid_op,
                  var_x,
                  # attrs
                  num_out_channels,
                  kernel_shape,
                  strides,
                  paddings,
                  dilations,
                  num_groups,
                  param_attr,
                  name_attr,
              ))
    var_conv = (name + '.conv') if var_b else var_y  # hidden variable
    prog.OpDesc(
        fluid_op,
        (['Input', 'Filter'], [var_x, var_w]),  # , 'Bias', 'ResidualData'
        (['Output'], [var_conv]),
        {
            'strides': strides,
            'paddings': paddings,
            'dilations': dilations,
            'groups': num_groups,
        },
    )
    if var_b:
        prog.VarDesc(var_conv)
        prog.IntermediateOp(
            '',
            'Add',
            [var_conv, var_b],  #
            [var_y],
            {'axis': 1},
            value_infos=value_infos,
            name=(name + '.bias'),
        )
    else:
        prog.VarDesc(var_y)


def ConvTranspose(prog,
                  inputs,
                  outputs,
                  attrs,
                  value_infos,
                  name,
                  embed_params=False,
                  *args,
                  **kwargs):
    """
    onnx::ConvTranspose-1:
    """

    # I/O
    var_x, var_w, var_b, = (inputs + [None] * 1)[:3]
    var_y, = outputs

    # interpretation
    assert attrs.get(
        'auto_pad',
        'NOTSET') == 'NOTSET', 'only auto_pad = NOTSET supported'  # optional
    assert sum(
        attrs.get('output_padding',
                  [])) == 0, 'only zero output_padding supported'  # optional ?
    kernel_shape = attrs.get('kernel_shape',
                             _shape(value_infos, var_w)[2:])  # optional, HW
    assert kernel_shape, 'kernel_shape not inferred'
    convnd = len(kernel_shape)
    assert 2 <= convnd <= 3, 'only conv2d_transpose and conv3d_transpose supported'
    num_out_channels = _shape(value_infos, var_w)[1]  # IO...

    fluid_op = 'conv{}d_transpose'.format(convnd)
    num_groups = attrs.get('group', 1)  # optional
    strides = attrs.get('strides', [1] * convnd)  # optional
    dilations = attrs.get('dilations', [1] * convnd)  # optional
    output_size = attrs.get('output_shape', [])  # optional
    pads = attrs.get('pads', [0] * (convnd * 2))  # optional
    paddings, var_x = _pad_if_asymmetric(prog, pads, var_x, value_infos)
    name_attr = ', name={}'.format(repr(name))
    if embed_params:
        embed_params = _check_embeddable(
            value_infos, *([var_w] + ([var_b] if var_b else [])))
        if not embed_params:
            _logger.warning('for op  %s(%s -> ConvTranspose -> %s)', name,
                            inputs, outputs)
            _logger.warning('broken Python code will be generated')
    if embed_params:
        embedded_w = name + '.w_0'
        value_infos[var_w]['embedded_as'].append(embedded_w)
        var_w = embedded_w
        if var_b:
            embedded_b = name + '.b_0'
            value_infos[var_b]['embedded_as'].append(embedded_b)
            var_b = embedded_b
            param_attr = ''
        else:
            param_attr = ', bias_attr=False'
    else:
        param_attr = ', param_attr={}, bias_attr={}'.format(
            repr(var_w),
            repr(var_b) if var_b else False)

    # generation
    prog.Code('{} = layers.{}({}'
              ', num_filters={}'
              ', output_size={}'
              ', filter_size={}'
              ', padding={}'
              ', stride={}'
              ', dilation={}'
              ', groups={}'
              '{}{})'.format(
                  var_y,
                  fluid_op,
                  var_x,
                  # attrs
                  num_out_channels,
                  output_size or None,
                  kernel_shape,
                  paddings,
                  strides,
                  dilations,
                  num_groups,
                  param_attr,
                  name_attr,
              ))
    var_conv = (name + '.conv') if var_b else var_y  # hidden variable
    prog.OpDesc(
        fluid_op,
        (['Input', 'Filter'], [var_x, var_w]),  # , 'Bias', 'ResidualData'
        (['Output'], [var_conv]),
        {
            'strides': strides,
            'paddings': paddings,
            'dilations': dilations,
            'groups': num_groups,
            # unused
            'output_size': output_size,
        },
    )
    if var_b:
        prog.VarDesc(var_conv)
        prog.IntermediateOp(
            '',
            'Add',
            [var_conv, var_b],  #
            [var_y],
            {'axis': 1},
            value_infos=value_infos,
            name=(name + '.bias'),
        )
    else:
        prog.VarDesc(var_y)


def Gemm(prog, inputs, outputs, attrs, value_infos, name, *args, **kwargs):
    """
    onnx::Gemm-9:
    """

    # due to fluid fc don't support transposed weight, we use matmul + ew_add
    var_a, var_b, var_c, = inputs
    var_y, = outputs

    alpha = attrs.get('alpha', 1.)  # optional
    beta = attrs.get('beta', 1.)  # optional
    trans_a = bool(attrs.get('transA', 0))  # optional
    trans_b = bool(attrs.get('transB', 0))  # optional

    var_mm = var_y if beta == 0 else (name + '_mmed')  # explicit variable
    prog.Op(
        '',
        'MatMul',
        [var_a, var_b],
        [var_mm],
        {
            'transpose_x': trans_a,
            'transpose_y': trans_b,
            'alpha': alpha,
        },
        value_infos=value_infos,
        name=(name + '_mm'),
    )
    prog.op_descs[-1].attrs.extend(
        prog.OpDescAttrs({
            'transpose_X': trans_a,
            'transpose_Y': trans_b,
        }))  # f**k you API
    if beta != 0:
        if beta == 1.:  # exactly
            prog.Op(
                '',
                'Add',
                [var_mm, var_c],
                [var_y],
                {'axis': 1},
                value_infos=value_infos,
                name=(name + '_bias'),
            )
        else:
            var_beta = name + '_beta'  # explicit variable
            var_vm = name + '_vm'  # explicit variable
            if beta.is_integer():
                vm_dtype = _dtype_or_none(value_infos, var_c)
                if vm_dtype is None:
                    vm_dtype = _np.dtype('float32')
                    _logger.warning(
                        'in op %s(%s -> Gemm -> %s): '
                        'attribute "beta" seems to be an interger, '
                        'however dtype can not be inferred, '
                        'still use float32', name, inputs, outputs)
                beta = _np.dtype(vm_dtype).type(beta)
            prog.Op(
                '',
                'Constant',
                [],
                [var_beta],
                {'value': beta},
                value_infos=value_infos,
                name=var_beta,
            )
            prog.Op(
                '',
                'Mul',
                [var_c, var_beta],
                [var_vm],
                dict(),
                value_infos=value_infos,
                name=(var_beta + '_scale'),
            )
            prog.Op(
                '',
                'Add',
                [var_mm, var_vm],
                [var_y],
                {'axis': 1},  #
                name=(name + '_bias'),
            )


def GlobalAveragePool(prog,
                      inputs,
                      outputs,
                      attrs,
                      value_infos,
                      name='',
                      *args,
                      **kwargs):
    """
    onnx::GlobalAveragePool-1:
    """

    return _global_pool(prog,
                        'avg',
                        inputs,
                        outputs,
                        attrs,
                        value_infos,
                        name=name)


def GlobalMaxPool(prog,
                  inputs,
                  outputs,
                  attrs,
                  value_infos,
                  name='',
                  *args,
                  **kwargs):
    """
    onnx::GlobalMaxPool-1:
    """

    return _global_pool(prog,
                        'max',
                        inputs,
                        outputs,
                        attrs,
                        value_infos,
                        name=name)


def GRU(prog, inputs, outputs, attrs, value_infos, *args, **kwargs):
    """
    onnx::GRU-7:
    """

    var_x, var_w, var_r, var_b, var_len, var_xh, = (inputs + [None] * 3)[:6]
    var_y, var_yh, = (outputs + [None] * 2)[:2]
    var_gate = var_y + '.gate'  # dummy output
    var_reset = var_y + '.reset'  # dummy output
    var_hidden = var_y + '.hidden'  # dummy output, # var_yh

    # interpretation
    x_shape = _shape_or_none(value_infos, var_x)
    assert x_shape is not None, 'shape of X required to be known'
    assert x_shape[1] == 1, 'only X with batch_size = 1 supported'
    assert 'clip' not in attrs, 'clipping not supported'
    hidden_size = attrs.get('hidden_size', None)  # optional
    if not hidden_size:
        r_shape = _shape_or_none(value_infos, var_r)
        if r_shape:
            hidden_size = r_shape[-1]
    if not hidden_size:
        w_shape = _shape_or_none(value_infos, var_w)
        if w_shape:
            hidden_size = w_shape[-2] // 3
    if not hidden_size and var_b:
        b_shape = _shape_or_none(value_infos, var_b)
        if b_shape:
            hidden_size = b_shape[-1] // 6
    if not hidden_size and var_xh:
        xh_shape = _shape_or_none(value_infos, var_xh)
        if xh_shape:
            hidden_size = xh_shape[-1]
    assert hidden_size, 'hidden_size not inferred'
    assert attrs.get(
        'linear_before_reset',
        0) == 0, 'only linear_before_reset = 0 supported'  # optional
    direction = attrs.get('direction', 'forward')  # optional
    assert direction != 'bidirectional', 'direction = bidirectional not supported'
    activations = attrs.get('activations', ['Sigmoid', 'Tanh'])  # optional
    assert len(activations) == 2, 'bidirectional operation not supported'
    activations = [s.lower() for s in activations]  # TODO: check support
    gate_activation, candidate_activation = activations
    is_reverse = direction == 'reverse'

    fluid_op = 'dynamic_gru'

    # generation
    var_x0 = var_x + '_0'  # explicit variable
    prog.Op(
        '',
        'Squeeze',
        [var_x],
        [var_x0],
        {'axes': [1]},  # index on n
        name=(var_x + '_index'),
    )
    var_w0 = var_w + '_0'  # explicit variable
    prog.Op(
        '',
        'Squeeze',
        [var_w],
        [var_w0],
        {'axes': [0]},  # index on d
        name=(var_w + '_index'),
    )
    var_fc = var_x0 + '_fc'
    var_mm = (var_x0 + '_mmed') if var_b else var_fc
    prog.Op(
        '',
        'MatMul',
        [var_x0, var_w0],
        [var_mm],
        {
            'transpose_x': 0,
            'transpose_y': 1,
        },
        value_infos=value_infos,
        name=(var_x0 + '_mm'),
    )
    prog.op_descs[-1].attrs.extend(
        prog.OpDescAttrs({
            'transpose_X': 0,
            'transpose_Y': 1,
        }))  # f**k you API
    var_r0 = var_r + '_0'  # explicit variable
    prog.Op(
        '',
        'Squeeze',
        [var_r],
        [var_r0],
        {'axes': [0]},  # index on d
        name=(var_r + '_index'),
    )
    var_r0t = var_r0 + '_t'  # explicit variable
    prog.Op(
        '',
        'Transpose',
        [var_r0],
        [var_r0t],
        {'perm': [1, 0]},  # transpose OI->IO
        name=(var_r0 + '_transpose'),
    )
    if var_b:
        var_bi = var_b + '_i'  # explicit variable
        var_bh = var_b + '_h'  # explicit variable
        prog.Op(
            '',
            'Split',
            [var_b],
            [var_bi, var_bh],
            {
                'axis': 1,  # split on x
                'split': [hidden_size * 3, hidden_size * 3],
            },
            name=(var_b + '_split'),
        )
        # squeeze bi so Gemm Add can be performed on axis=1 exaclty
        var_bi0 = var_bi + '_0'  # explicit variable
        prog.Op(
            '',
            'Squeeze',
            [var_bi],
            [var_bi0],
            {'axes': [0]},  # slice on d
            name=(var_bi + '_index'),
        )
        prog.Op(
            '',
            'Add',
            [var_mm, var_bi0],
            [var_fc],
            {'axis': 1},  #
            name=(var_x0 + '_bias'),
        )
    if var_xh:
        var_xh0 = var_xh + '_0'  # explicit variable
        prog.Op(
            '',
            'Squeeze',
            [var_xh],
            [var_xh0],
            {'axes': [1]},  # index on n
            name=(var_xh + '_index'),
        )
    var_y00 = var_y + '_00'  # explicit variable
    prog.Code('{} = layers.{}({}, {}, origin_mode=True'
              ', h_0={}'
              ', is_reverse={}'
              ', gate_activation={}'
              ', candidate_activation={}'
              ', param_attr={}, bias_attr={})'.format(
                  var_y00,
                  fluid_op,
                  var_fc,
                  hidden_size,
                  var_xh0 if var_xh else None,
                  is_reverse,
                  repr(gate_activation),
                  repr(candidate_activation),
                  repr(var_r0t),
                  repr(var_bh) if var_b else False,
              ))

    fluid_op = 'gru'
    prog.VarDesc(var_y00)
    prog.VarDesc(var_gate)
    prog.VarDesc(var_reset)
    prog.VarDesc(var_hidden)
    prog.OpDesc(
        fluid_op,
        (['Input', 'Weight', 'Bias', 'H0'], [var_fc, var_r0t] +
         ([var_bh] if var_b else []) + ([var_xh0] if var_xh else [])),
        (['Hidden', 'BatchGate', 'BatchResetHiddenPrev', 'BatchHidden'
          ], [var_y00, var_gate, var_reset, var_hidden]),
        {
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'activation': candidate_activation,
            'origin_mode': True,
        },
    )
    prog.Op(
        '',
        'Unsqueeze',
        [var_y00],
        [var_y],
        {'axes': [1, 1]},  # extrude on dn
        name=(var_y + '_reshape'),
    )


def LSTM(prog, inputs, outputs, attrs, value_infos, name, *args, **kwargs):
    """
    onnx::LSTM-7:
    """

    var_x, var_w, var_r, var_b, var_len, var_xh, var_xc, var_p, = (
        inputs + [None] * 5)[:8]
    var_y, var_yh, var_yc, = (outputs + [None] * 3)[:3]
    var_gate = name + '.gate'
    var_pre = name + '.pre'

    # interpretation
    x_shape = _shape_or_none(value_infos, var_x)
    assert x_shape is not None, 'shape of X required to be known'
    assert x_shape[1] == 1, 'only X with batch_size = 1 supported'
    assert 'clip' not in attrs, 'clipping not supported'
    hidden_size = attrs.get('hidden_size', None)  # optional
    if not hidden_size:
        r_shape = _shape_or_none(value_infos, var_r)
        if r_shape:
            hidden_size = r_shape[-1]
    if not hidden_size:
        w_shape = _shape_or_none(value_infos, var_w)
        if w_shape:
            hidden_size = w_shape[-2] // 4
    if not hidden_size and var_b:
        b_shape = _shape_or_none(value_infos, var_b)
        if b_shape:
            hidden_size = b_shape[-1] // 8
    if not hidden_size and var_xh:
        xh_shape = _shape_or_none(value_infos, var_xh)
        if xh_shape:
            hidden_size = xh_shape[-1]
    if not hidden_size and var_xc:
        xc_shape = _shape_or_none(value_infos, var_xc)
        if xc_shape:
            hidden_size = xc_shape[-1]
    if not hidden_size and var_p:
        p_shape = _shape_or_none(value_infos, var_p)
        if p_shape:
            hidden_size = p_shape[-1] // 3
    assert hidden_size, 'hidden_size not inferred'
    assert attrs.get(
        'linear_before_reset',
        0) == 0, 'only linear_before_reset = 0 supported'  # optional
    assert attrs.get('input_forget',
                     0) == 0, 'only input_forget = 0 supported'  # optional
    direction = attrs.get('direction', 'forward')  # optional
    assert direction != 'bidirectional', 'direction = bidirectional not supported'
    activations = attrs.get('activations',
                            ['Sigmoid', 'Tanh', 'Tanh'])  # optional
    assert len(activations) == 3, 'bidirectional operation not supported'
    activations = [s.lower() for s in activations]  # TODO: check support
    gate_activation, cell_activation, candidate_activation = activations
    is_reverse = direction == 'reverse'

    fluid_op = 'dynamic_lstm'
    name_attr = ', name={}'.format(repr(name))

    # generation
    var_x0 = var_x + '_0'  # explicit variable
    prog.Op(
        '',
        'Squeeze',
        [var_x],
        [var_x0],
        {'axes': [1]},  # index on n
        name=(var_x + '_index'),
    )
    var_w0 = var_w + '_0'  # explicit variable
    prog.Op(
        '',
        'Squeeze',
        [var_w],
        [var_w0],
        {'axes': [0]},  # index on d
        name=(var_w + '_index'),
    )
    var_fc = var_x0 + '_fc'
    var_mm = (var_x0 + '_mmed') if var_b else var_fc
    prog.Op(
        '',
        'MatMul',
        [var_x0, var_w0],
        [var_mm],
        {
            'transpose_x': 0,
            'transpose_y': 1,
        },
        value_infos=value_infos,
        name=(name + '_mm'),
    )
    prog.op_descs[-1].attrs.extend(
        prog.OpDescAttrs({
            'transpose_X': 0,
            'transpose_Y': 1,
        }))  # f**k you API
    var_r0 = var_r + '_0'  # explicit variable
    prog.Op(
        '',
        'Squeeze',
        [var_r],
        [var_r0],
        {'axes': [0]},  # index on d
        name=(var_r + '_index'),
    )
    var_r0t = var_r0 + '_t'  # explicit variable
    prog.Op(
        '',
        'Transpose',
        [var_r0],
        [var_r0t],
        {'perm': [1, 0]},  # transpose OI->IO
        name=(var_r0 + '_transpose'),
    )
    if var_b:
        var_bi = var_b + '_i'  # explicit variable
        var_bh = var_b + '_h'  # explicit variable
        prog.Op(
            '',
            'Split',
            [var_b],
            [var_bi, var_bh],
            {
                'axis': 1,  # split on x
                'split': [hidden_size * 4, hidden_size * 4],
            },
            name=(var_b + '_split'),
        )
        # squeeze bi so Gemm Add can be performed on axis=1 exaclty
        var_bi0 = var_bi + '_0'  # explicit variable
        prog.Op(
            '',
            'Squeeze',
            [var_bi],
            [var_bi0],
            {'axes': [0]},  # slice on d
            name=(var_bi + '_index'),
        )
        prog.Op(
            '',
            'Add',
            [var_mm, var_bi0],
            [var_fc],
            {'axis': 1},  #
            name=(name + '_bias'),
        )
    if var_xh:
        var_xh0 = var_xh + '_0'  # explicit variable
        prog.Op(
            '',
            'Squeeze',
            [var_xh],
            [var_xh0],
            {'axes': [1]},  # index on n
            name=(var_xh + '_index'),
        )
    if var_xc:
        var_xc0 = var_xc + '_0'  # explicit variable
        prog.Op(
            '',
            'Squeeze',
            [var_xc],
            [var_xc0],
            {'axes': [1]},  # index on n
            name=(var_xc + '_index'),
        )
    var_bhp = var_p
    if var_b:
        if var_p:
            var_bhp = var_bh + '_p'  # explicit variable
            prog.Op(
                '',
                'Concat',
                [var_bh, var_p],
                [var_bhp],
                {'axes': [1]},  # cat on x
                name=(name + '_concat'),
            )
        else:
            var_bhp = var_bh
    var_yh0 = var_yh + '_0'  # explicit variable
    var_yc0 = var_yc + '_0'  # explicit variable
    prog.Code('{}, {} = layers.{}({}, {}'
              ', h_0={}'
              ', c_0={}'
              ', use_peepholes={}'
              ', is_reverse={}'
              ', gate_activation={}'
              ', cell_activation={}'
              ', candidate_activation={}'
              ', param_attr={}, bias_attr={}'
              '{})'.format(
                  var_yh0,
                  var_yc0,
                  fluid_op,
                  var_fc,
                  hidden_size * 4,
                  var_xh0 if var_xh else None,
                  var_xc0 if var_xc else None,
                  bool(var_p),
                  is_reverse,
                  repr(gate_activation),
                  repr(cell_activation),
                  repr(candidate_activation),
                  repr(var_r0t),
                  repr(var_bhp) if var_bhp else False,
                  name_attr,
              ))

    fluid_op = 'lstm'
    prog.VarDesc(var_yh0)
    prog.VarDesc(var_yc0)
    prog.VarDesc(var_gate)
    prog.VarDesc(var_pre)
    prog.OpDesc(
        fluid_op,
        (['Input', 'Weight', 'Bias', 'H0', 'C0'], [var_fc, var_r0t] +
         ([var_bhp] if var_bhp else []) + ([var_xh0] if var_xh else []) +
         ([var_xc0] if var_xc else [])),
        (['Hidden', 'Cell', 'BatchGate', 'BatchCellPreAct'
          ], [var_yh0, var_yc0, var_gate, var_pre]),
        {
            'use_peepholes': bool(var_p),
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'cell_activation': cell_activation,
            'candidate_activation': candidate_activation,
        },
    )
    #    if var_yh:
    prog.Op(
        '',
        'Unsqueeze',
        [var_yh0],
        [var_y],  # var_yh
        {'axes': [1, 1]},  # extrude on dn
        name=(var_y + '_reshape'),
    )
    if var_yc:
        prog.Op(
            '',
            'Unsqueeze',
            [var_yc0],
            [var_yc],
            {'axes': [1, 1]},  # extrude on dn
            name=(var_yc + '_reshape'),
        )


def MaxPool(prog, inputs, outputs, attrs, value_infos, name='', *args,
            **kwargs):
    """
    onnx::MaxPool-10:
    """

    return _pool(prog, 'max', inputs, outputs, attrs, value_infos, name=name)


def MaxRoiPool(prog, inputs, outputs, attrs, value_infos, name, *args,
               **kwargs):
    """
    onnx::MaxRoiPool-1:
    """

    _roi_pool(prog, 'roi_pool', inputs, outputs, attrs, value_infos, name)


def Pad(prog, inputs, outputs, attrs, value_infos, name='', *args, **kwargs):
    """
    onnx::Pad-2:
    """

    # I/O
    var_data, = inputs
    var_output, = outputs

    # interpretation
    pads = attrs['pads']  # required
    mode = attrs.get('mode', 'constant')  # optional
    value = attrs.get('value', 0.)  # optional
    data_shape = _shape_or_none(value_infos, var_data)
    output_shape = _shape_or_none(value_infos, var_output)
    assume_pad2d = False
    if len(pads) == 4:
        assume_pad2d |= mode != 'constant'
        if data_shape is not None:
            assume_pad2d |= data_shape and len(data_shape) == 4  # NCHW
        if output_shape is not None:
            assume_pad2d |= output_shape and len(output_shape) == 4  # NCHW
    od_attrs = {'pad_value': value}
    if assume_pad2d:
        fluid_op = 'pad2d'
        pad2d_attr = ', mode={}, data_format="NCHW"'.format(repr(mode))
        od_attrs['mode'] = mode
        od_attrs['data_format'] = "NCHW"
    else:
        assert mode == 'constant', 'mode {} supported only in pad2d'.format(
            mode)
        fluid_op = 'pad'
        pad2d_attr = ''
    paddings = _np.array(pads).reshape(
        (-1, 2)).transpose().flatten().tolist()  # SSEE -> SESE
    od_attrs['paddings'] = paddings
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('{} = layers.{}({}'
              ', paddings={}'
              ', pad_value={}'
              '{}{})'.format(
                  var_output,
                  fluid_op,
                  var_data,
                  # attrs
                  paddings,
                  value,
                  pad2d_attr,
                  name_attr,
              ))
    prog.VarDesc(var_output)
    prog.OpDesc(
        fluid_op,
        (['X'], [var_data]),
        (['Out'], [var_output]),
        od_attrs,
    )


def PRelu(prog,
          inputs,
          outputs,
          attrs,
          value_infos,
          name='',
          embed_params=False,
          *args,
          **kwargs):
    """
    onnx::PRelu-9:
    """

    # I/O
    var_x, var_slope, = inputs
    var_y, = outputs

    # interpretation
    mode = 'channel'
    slope_shape = _shape_or_none(value_infos, var_slope)
    if slope_shape is not None:
        if len(slope_shape) == 0:
            mode = 'all'
        elif len(slope_shape) >= 2:
            if slope_shape[1] != _np.product(
                    slope_shape):  # not channel broadcasting
                mode = 'element'
    fluid_op = 'prelu'
    name_attr = ', name={}'.format(repr(name)) if name else ''
    if embed_params:
        embed_params = _check_embeddable(value_infos, var_slope)
        if not embed_params and name:
            _logger.warning('for op  %s(%s -> PRelu -> %s)', name, inputs,
                            outputs)
            _logger.warning('broken Python code will be generated')
    if embed_params:
        assert name != ''
        embedded_slope = name + '.w_0'
        value_infos[var_slope]['embedded_as'].append(embedded_slope)
        var_slope = embedded_slope
        param_attr = ''
    else:
        param_attr = ', param_attr={}'.format(repr(var_slope))

    # generation
    prog.Code('{} = layers.{}({}'
              ', mode={}'
              '{}{})'.format(
                  var_y,
                  fluid_op,
                  var_x,
                  # attrs
                  repr(mode),
                  param_attr,
                  name_attr,
              ))
    prog.VarDesc(var_y)
    prog.OpDesc(
        fluid_op,
        (['X', 'Alpha'], [var_x, var_slope]),
        (['Out'], [var_y]),
        {'mode': mode},
    )


def PsRoiPool(prog, inputs, outputs, attrs, value_infos, name, *args, **kwargs):
    """
    caffe2::PsRoiPool
    """

    _roi_pool(prog, 'psroi_pool', inputs, outputs, attrs, value_infos, name)


def Reshape(prog, inputs, outputs, attrs, value_infos, name, *args, **kwargs):
    """
    onnx::Reshape-5:
    """

    # I/O
    var_data, var_shape, = inputs
    var_reshaped, = outputs

    # interpretation
    shape = _const_weight_or_none(value_infos, var_shape)
    is_const_shape = shape and 'const_value' in value_infos[var_shape]
    if shape is None:
        shape = _shape_or_none(value_infos, var_reshaped)


#    assert shape is not None, ('given shape is neither const value nor deductible from output, '
#                               'this is not supported')
    if shape is None:
        shape = [1, -1]  # who knows
        _logger.warning(
            'in op %s(%s -> Reshape -> %s): '
            'input "shape" not inferred, use [1, -1] as dummy value, '
            'the behavior of Paddle fluid maybe undefined', name, inputs,
            outputs)
    fluid_op = 'reshape'
    name_attr = ', name={}'.format(repr(name))

    # generation
    var_shape_int32 = var_shape + '_int32'  # explicit variable
    prog.Code('# shape:{}={} # const as literal'.format(var_shape, shape))
    if is_const_shape:
        prog.Code('{} = layers.{}({}'
                  ', shape={}'
                  '{})'.format(
                      var_reshaped,
                      fluid_op,
                      var_data,
                      # attrs
                      shape,
                      name_attr,
                  ))
    else:
        prog.Op(
            '',
            'Cast',
            [var_shape],
            [var_shape_int32],
            {'to': _np.dtype('int32')},  # use np.dtype
            value_infos=value_infos,
            name=(name + '_cast'),
        )
        prog.Code('{} = layers.{}({}'
                  ', shape={}'
                  ', actual_shape={}'
                  '{})'.format(
                      var_reshaped,
                      fluid_op,
                      var_data,
                      # attrs
                      shape,
                      var_shape_int32,
                      name_attr,
                  ))
    fluid_op = 'reshape2'
    var_xshape = name + '.xshape'  # dummy output
    prog.VarDesc(var_reshaped)
    prog.VarDesc(var_xshape)
    prog.OpDesc(
        fluid_op,
        (['X', 'Shape'], [var_data, var_shape_int32]),
        (['Out', 'XShape'], [var_reshaped, var_xshape]),
        {'shape': shape},
    )


def Resize(prog, inputs, outputs, attrs, value_infos, name='', *args, **kwargs):
    """
    onnx::Resize-10:
    """

    return _interpolate(prog, inputs, outputs, attrs, value_infos, name=name)


def RoiAlign(prog, inputs, outputs, attrs, value_infos, name, *args, **kwargs):
    """
    caffe2::RoiAlign
    """

    _roi_pool(prog, 'roi_align', inputs, outputs, attrs, value_infos, name)


#def Shape(
#        prog, inputs, outputs, attrs, value_infos,
#        *args, **kwargs):
#    """
#    onnx::ConstantOfShape-1:
#    """
#
#    # I/O
#    val_data, = inputs
#    val_shape, = outputs
#    var_data = _make_var_name(val_data)
#    var_shape = _make_var_name(val_shape)
#
#    # interpretation
#    fluid_op = 'shape'
##    value_infos[val_shape]['remove_batch'] = False
#
#    # generation
#    prog.Code('{} = layers.{}({})'
#              .format(var_shape,
#                      fluid_op,
#                      var_data,
#                      # attrs
#                      ))
#    prog.VarDesc(var_shape) # , _value_info_or_none(value_infos, val_shape))
#    prog.OpDesc(fluid_op,
#                ([var_data], 'X'),
#                ([var_shape], 'Out'),
#                dict(),
#                )


def Slice(prog, inputs, outputs, attrs, value_infos, *args, **kwargs):
    """
    onnx::Slice-1:9
    """

    # I/O
    var_data, = inputs
    var_output, = outputs

    # interpretation
    fluid_op = 'slice'
    axes = attrs['axes']  # required
    starts = attrs['starts']  # required
    ends = attrs['ends']  # required
    shape = _shape_or_none(value_infos, var_data)
    if shape is not None:
        #        ndims = len(shape)
        #        for idx, value in enumerate(axes):
        #            if value > ONNX_INT_MAX // 2:
        #                axes[idx] = ndims + value - ONNX_INT_MAX
        #  FIXME: Paddle 1.3 Doc: '对于未知大小维度的末尾进行切片，则建议传入 INT_MAX' not works ?
        for idx, value in enumerate(starts):
            if value > ONNX_INT_MAX // 2:
                value = value - ONNX_INT_MAX
                starts[idx] = shape[axes[idx]] + value
        for idx, value in enumerate(ends):
            if value > ONNX_INT_MAX // 2:
                value = value - ONNX_INT_MAX
                ends[idx] = shape[axes[idx]] + value

    # generation
    prog.Code('{} = layers.{}({}'
              ', axes={}'
              ', starts={}'
              ', ends={}'
              ')'.format(
                  var_output,
                  fluid_op,
                  var_data,
                  # attrs
                  axes,
                  starts,
                  ends,
              ))
    prog.VarDesc(var_output)
    prog.OpDesc(
        fluid_op,
        (['Input'], [var_data]),
        (['Out'], [var_output]),
        {
            'axes': axes,
            'starts': starts,
            'ends': ends,
        },
    )


def Split(prog, inputs, outputs, attrs, *args, name='', **kwargs):
    """
    onnx::Split-2:
    """

    # I/O
    var_input, = inputs

    # interpretation
    fluid_op = 'split'
    split = attrs['split']  # required
    axis = attrs.get('axis', 0)  # optional
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('{} = layers.{}({}, {}'
              ', dim={}'
              '{})'.format(
                  ', '.join(outputs),
                  fluid_op,
                  var_input,
                  split,
                  # attrs
                  axis,
                  name_attr,
              ))
    for var_out in outputs:
        prog.VarDesc(var_out)
    prog.OpDesc(
        fluid_op,
        (['X'], [var_input]),
        (['Out'] * len(outputs), outputs),
        {
            'axis': axis,
            'sections': split,
            # unused
            'num': 0,
        },
    )


def Sum(prog, inputs, outputs, *args, **kwargs):
    """
    onnx::Sum-8:
    """

    # I/O
    var_sum, = outputs

    # interpretation
    fluid_op = 'sums'

    # generation
    prog.Code('{} = layers.{}({})'.format(
        var_sum,
        fluid_op,
        '[' + ', '.join(inputs) + ']',
        # attrs
    ))
    fluid_op = 'sum'
    prog.VarDesc(var_sum)
    prog.OpDesc(
        fluid_op,
        (['X'] * len(inputs), inputs),
        (['Out'], [var_sum]),
        dict(),
    )


def Tile(prog, inputs, outputs, attrs, value_infos, name='', *args, **kwargs):
    """
    onnx::Tile-1:
    """

    # I/O
    var_input, var_repeats, = inputs
    var_output, = outputs

    # interpretation
    repeats = _const_weight_or_none(value_infos, var_repeats)
    assert repeats is not None, 'only const repeats supported'
    fluid_op = 'expand'
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('# repeats:{}={} # const as literal'.format(var_repeats, repeats))
    prog.Code('{} = layers.{}({}'
              ', expand_times={}'
              '{})'.format(
                  var_output,
                  fluid_op,
                  var_input,
                  # attrs
                  repeats,
                  name_attr,
              ))
    prog.VarDesc(var_output)
    prog.OpDesc(
        fluid_op,
        (['X'], [var_input]),
        (['Out'], [var_output]),
        {'expand_times': repeats},
    )


def Transpose(prog, inputs, outputs, attrs, *args, name='', **kwargs):
    """
    onnx::Transpose-1:
    """

    # I/O
    var_data, = inputs
    var_transposed, = outputs

    # interpretation
    fluid_op = 'transpose'
    perm = attrs['perm']  # required
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('{} = layers.{}({}'
              ', perm={}'
              '{})'.format(
                  var_transposed,
                  fluid_op,
                  var_data,
                  # attrs
                  perm,
                  name_attr,
              ))
    fluid_op = 'transpose2'
    var_xshape = name + '.xshape'  # dummy output
    prog.VarDesc(var_xshape)
    prog.VarDesc(var_transposed)
    prog.OpDesc(
        fluid_op,
        (['X'], [var_data]),
        (['Out', 'XShape'], [var_transposed, var_xshape]),
        {'axis': perm},  # f**k you API
    )


def Upsample(prog,
             inputs,
             outputs,
             attrs,
             value_infos,
             name='',
             *args,
             **kwargs):
    """
    onnx::Upsample-9:9
    """

    return _interpolate(prog, inputs, outputs, attrs, value_infos, name=name)


if __name__ == '__main__':
    _logging.basicConfig(
        format=
        '[%(levelname)8s]%(name)s::%(funcName)s:%(lineno)04d: %(message)s',
        level=_logging.DEBUG,
    )
    logger = _logging.getLogger('symbolic_test')

    import numpy as np

    from onnx2fluid.writer import Program

    prog = Program()
    AdaptiveAveragePool(
        prog,
        ['X'],
        ['Y'],
        dict(output_size=[3, 3]),
        dict(Y=dict(shape=(2, 3, 3, 3), dtype=np.float32)),
        name='AdaptiveAveragePool2d',
    )
    logger.info('AdaptiveAveragePool2d program:\n%s', prog)

    prog = Program()
    AdaptiveAveragePool(
        prog,
        ['X'],
        ['Y'],
        dict(output_size=[3, 3, 3]),
        dict(Y=dict(shape=(2, 3, 3, 3, 3), dtype=np.float32)),
        name='AdaptiveAveragePool3d',
    )
    logger.info('AdaptiveAveragePool3d program:\n%s', prog)

    prog = Program()
    AffineGrid(
        prog,
        ['Theta'],
        ['Grid'],
        dict(size=[2, 2, 8, 8]),
        dict(Grid=dict(shape=(2, 8, 8, 2), dtype=np.float32)),
    )
    logger.info('AffineGrid program:\n%s', prog)

    prog = Program()
    BatchNormalization(
        prog,
        ['X', 'scale', 'B', 'mean', 'var'],
        ['Y'],
        dict(
            epsilon=1e-5,
            momentum=.9,
        ),
        dict(
            scale=dict(shape=(3, ), dtype=np.float32),
            B=dict(shape=(3, ), dtype=np.float32),
            mean=dict(shape=(3, ), dtype=np.float32),
            var=dict(shape=(3, ), dtype=np.float32),
            Y=dict(shape=(2, 3), dtype=np.float32),
        ),
        name='BatchNormalization',
        embed_params=True,
    )
    logger.info('BatchNormalization program:\n%s', prog)

    prog = Program()
    Cast(
        prog,
        ['input'],
        ['output'],
        dict(to=2),  # TensorProto.UINT8
        dict(input=dict(shape=(2, 3), dtype=np.float32),
             output=dict(shape=(2, 3), dtype=np.uint8)),
    )
    logger.info('Cast program:\n%s', prog)

    prog = Program()
    _default(
        prog,
        'Clip',
        ['input'],
        ['output'],
        dict(min=-1., max=1.),
        dict(output=dict(shape=(2, 3), dtype=np.float32)),
    )
    logger.info('Clip program:\n%s', prog)

    prog = Program()
    Conv(
        prog,
        ['X', 'W'],
        ['Y'],
        dict(
            auto_pad='NOTSET',
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        dict(
            W=dict(shape=(2, 3, 3, 3), dtype=np.float32),
            Y=dict(shape=(2, 2, 4, 6), dtype=np.float32),
        ),
        name='ConvNoBias2d',
        embed_params=True,
    )
    logger.info('ConvNoBias2d program:\n%s', prog)

    prog = Program()
    Conv(
        prog,
        ['X', 'W', 'B'],
        ['Y'],
        dict(
            auto_pad='NOTSET',
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        dict(
            W=dict(shape=(2, 3, 3, 3), dtype=np.float32),
            B=dict(shape=(2), dtype=np.float32),
            Y=dict(shape=(2, 2, 4, 6), dtype=np.float32),
        ),
        name='Conv2d',
        embed_params=True,
    )
    logger.info('Conv2d program:\n%s', prog)

    prog = Program()
    ConvTranspose(
        prog,
        ['X', 'W', 'B'],
        ['Y'],
        dict(
            auto_pad='NOTSET',
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
            #              output_padding=[1, 1, 1, 1],
            #              output_shape=[6, 8],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        dict(
            W=dict(shape=(2, 3, 3, 3), dtype=np.float32),
            B=dict(shape=(2), dtype=np.float32),
            Y=dict(shape=(2, 2, 6, 8), dtype=np.float32),
        ),
        name='ConvTransposed2d',
        embed_params=True,
    )
    logger.info('ConvTransposed2d program:\n%s', prog)

    prog = Program()
    Conv(
        prog,
        ['X', 'W'],
        ['Y'],
        dict(
            auto_pad='NOTSET',
            dilations=[1, 1, 1],
            group=1,
            kernel_shape=[3, 3, 3],
            pads=[1, 1, 1, 1, 1, 1],
            strides=[1, 1, 1],
        ),
        dict(
            W=dict(shape=(2, 3, 3, 3, 3), dtype=np.float32),
            Y=dict(shape=(2, 2, 4, 6, 8), dtype=np.float32),
        ),
        name='ConvNoBias3d',
        embed_params=True,
    )
    logger.info('ConvNoBias3d program:\n%s', prog)

    prog = Program()
    Conv(
        prog,
        ['X', 'W', 'B'],
        ['Y'],
        dict(
            auto_pad='NOTSET',
            dilations=[1, 1, 1],
            group=1,
            kernel_shape=[3, 3, 3],
            pads=[1, 1, 1, 1, 1, 1],
            strides=[1, 1, 1],
        ),
        dict(
            W=dict(shape=(2, 3, 3, 3, 3), dtype=np.float32),
            B=dict(shape=(2), dtype=np.float32),
            Y=dict(shape=(2, 2, 4, 6, 8), dtype=np.float32),
        ),
        name='Conv3d',
        embed_params=True,
    )
    logger.info('Conv3d program:\n%s', prog)

    prog = Program()
    ConvTranspose(
        prog,
        ['X', 'W', 'B'],
        ['Y'],
        dict(
            auto_pad='NOTSET',
            dilations=[1, 1, 1],
            group=1,
            kernel_shape=[3, 3, 3],
            #              output_padding=[1, 1, 1, 1],
            #              output_shape=[6, 8],
            pads=[1, 1, 1, 1, 1, 1],
            strides=[1, 1, 1],
        ),
        dict(
            W=dict(shape=(2, 3, 3, 3, 3), dtype=np.float32),
            B=dict(shape=(2), dtype=np.float32),
            Y=dict(shape=(2, 2, 6, 8, 9), dtype=np.float32),
        ),
        name='ConvTransposed3d',
        embed_params=True,
    )
    logger.info('ConvTransposed3d program:\n%s', prog)

    prog = Program()
    _default(
        prog,
        'Equal',
        ['A', 'B'],
        ['C'],
        dict(),
        dict(C=dict(shape=(2, 3), dtype=np.bool)),
    )
    logger.info('Equal program:\n%s', prog)

    prog = Program()
    Gemm(
        prog,
        ['A', 'B', 'C'],
        ['Y'],
        dict(
            alpha=1.,
            beta=1.,
            transA=0,
            transB=1,
        ),
        dict(
            B=dict(shape=(8, 3), dtype=np.float32),
            Y=dict(shape=(2, 8), dtype=np.float32),
        ),
        name='Gemm',
    )
    logger.info('Gemm program:\n%s', prog)

    prog = Program()
    _default(
        prog,
        'Less',
        ['A', 'B'],
        ['C'],
        dict(),
        dict(C=dict(shape=(2, 3), dtype=np.bool)),
    )
    logger.info('Less program:\n%s', prog)

    prog = Program()
    _default(prog,
             'MatMul', ['A', 'B'], ['Y'],
             dict(),
             dict(Y=dict(shape=(2, 8), dtype=np.float32)),
             name='MatMul')
    logger.info('MatMul program:\n%s', prog)

    prog = Program()
    _default(
        prog,
        'OneHot',
        ['indices', 'depth', 'values'],
        ['output'],
        dict(axis=-1),
        dict(output=dict(shape=(2, 8), dtype=np.float32)),
    )
    logger.info('OneHot program:\n%s', prog)

    prog = Program()
    Pad(
        prog,
        ['data'],
        ['output'],
        dict(
            mode='constant',
            pads=[0, 1],
            value=0.,
        ),
        dict(
            data=dict(shape=(2, 7), dtype=np.float32),
            output=dict(shape=(2, 8), dtype=np.float32),
        ),
        name='Pad',
    )
    logger.info('Pad program:\n%s', prog)

    prog = Program()
    Pad(
        prog,
        ['data'],
        ['output'],
        dict(
            mode='reflect',
            pads=[0, 1, 2, 3],
            value=0.,
        ),
        dict(
            data=dict(shape=(2, 3, 3, 3), dtype=np.float32),
            output=dict(shape=(2, 3, 5, 7), dtype=np.float32),
        ),
        name='Pad2d',
    )
    logger.info('Pad2d program:\n%s', prog)

    prog = Program()
    PRelu(
        prog,
        ['X', 'slope'],
        ['Y'],
        dict(),
        dict(Y=dict(shape=(2, 3), dtype=np.float32)),
        name='PRelu',
    )
    logger.info('PRelu program:\n%s', prog)

    prog = Program()
    Tile(prog, ['input', 'repeats'], ['output'],
         dict(),
         dict(repeats=dict(const_value=[1, 2]),
              output=dict(shape=(2, 2, 4), dtype=np.float32)),
         name='Tile')
    logger.info('Tile program:\n%s', prog)
