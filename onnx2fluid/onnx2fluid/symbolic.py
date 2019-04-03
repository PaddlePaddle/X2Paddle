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
import numpy as np

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

DEFAULT_OP_MAPPING = {
        ## nil ops ##
        'RandomUniform':
            ['uniform_random', [], ['Out'], dict(high='max', low='min'),
             dict(), None, None, False],
        'RandomNormal':
            ['gaussian_random', [], ['Out'], dict(scale='std'),
             dict(), None, None, False],
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
        'LRN': ['lrn', ['X'], ['Out', 'MidOut'], dict(size='n', bias='k')], #
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
    #        'Transpose': ['transpose', ['X'], ['Out']],
        'Unsqueeze': ['unsqueeze', ['X'], ['Out']], # attrs bypassed, FIXME: emit unsqueeze2
        ## binary ops ##
        'Add': ['elementwise_add', ['X', 'Y'], ['Out'], dict(), dict(axis=-1)],
    #        'AffineGrid': ['affine_grid', ['Theta'], ['Output'], dict(size='out_shape')],
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


def _make_var_name(name):
    """
    make a valid variable name in Python code
    """

    if name == '':
        return '_'
    if name[0].isdigit():
        return 'var_' + name
    for s in ' *?\\/-:':
        name = name.replace(s, '_')
    if name.startswith('_'):
        name = 'var' + name
    return name


#def _value_info_or_none(value_infos, val_name):
#    return value_infos.get(val_name, None)


def _dtype(value_infos, val_name):
    return np.dtype(value_infos[val_name]['dtype'])


def _dtype_or_none(value_infos, val_name):
    if val_name not in value_infos:
        return None
    value_info = value_infos[val_name]
    if 'dtype' not in value_info:
        return None
    return np.dtype(value_info['dtype'])


def _shape(value_infos, val_name):
    return list(value_infos[val_name]['shape'])


def _shape_or_none(value_infos, val_name):
    if val_name not in value_infos:
        return None
    value_info = value_infos[val_name]
    if 'shape' not in value_info:
        return None
    return list(value_info['shape'])


def _const_weight_or_none(value_infos, val_name):
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


def _default(prog, op_type, inputs, outputs, attrs, *args, name='', **kwargs):
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

    val_inps = inputs if input_perm is None else map(lambda i: inputs[i],
                                                     input_perm)
    val_outs = outputs if output_perm is None else map(lambda i: outputs[i],
                                                       output_perm)
    var_inps = [_make_var_name(val) for val in val_inps]
    var_outs = [_make_var_name(val) for val in val_outs]
    arg_name = ', name={}'.format(
        repr(name)) if fill_name_field and name else ''
    arg_attrs = [
        ', {}={}'.format(key, value) for key, value in fluid_attrs.items()
    ]

    prog.Code('{} = layers.{}({}{}{})'.format(
        ', '.join(var_outs),
        fluid_op,
        ', '.join(var_inps),
        ''.join(arg_attrs),
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
    prog.OpDesc(fluid_op, (var_inps, *fluid_input_args),
                (var_outs, *fluid_output_args), fluid_attrs)


def _assign(prog, attrs):
    mapping = attrs['mapping']  # additional
    fluid_op = 'assign'

    for val_dst, val_src in mapping.items():
        var_dst = _make_var_name(val_dst)
        var_src = _make_var_name(val_src)
        prog.Code('{} = {}'.format(var_dst, var_src))
        #        prog.Code('{} = layers.{}({})'
        #                  .format(var_dst,
        #                          fluid_op,
        #                          var_src,
        #                          ))
        prog.VarDesc(var_dst)
        prog.OpDesc(
            fluid_op,
            ([var_src], 'X'),
            ([var_dst], 'Out'),
            dict(),
        )


def _zeros_like(prog, val_ref, val_out, value_infos):
    prog.Op(
        '',
        'Sub',
        [val_ref, val_ref],
        [val_out],  # val
        dict(axis=0),
        value_infos,
    )


def _pad_if_asymmetric(prog, pads, val_name, value_infos):  # pads: SSEE
    assert len(pads) & 1 == 0
    ndims = len(pads) // 2
    symmetric = True
    for idx_dim in range(ndims):
        if pads[idx_dim] != pads[ndims + idx_dim]:
            symmetric = False
            break
    if symmetric:
        return pads[:ndims], val_name

    val_padded = val_name + '_padded'  # explicit variable
    prog.Op(
        '',
        'Pad',
        [val_name],
        [val_padded],  # val
        dict(
            mode='constant',
            value=0.,
            pads=pads,
        ),
        value_infos=value_infos,
        name=val_padded,
    )
    return [0] * ndims, val_padded


def _adaptive_pool(prog, pool_type, inputs, outputs, attrs, name=''):
    # I/O
    val_x, = inputs
    val_y, = outputs[:1]
    var_x = _make_var_name(val_x)
    var_y = _make_var_name(val_y)

    has_indices = len(outputs) > 1
    if has_indices:
        val_indices = outputs[1]
        var_indices = _make_var_name(val_indices)

    # interpretation
    pool_size = attrs['output_size']  # required
    poolnd = len(pool_size)
    assert 2 <= poolnd <= 3, 'only pool2d and pool3d is supported'

    fluid_op = 'adaptive_pool{}d'.format(poolnd)
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('{}{} = layers.{}({}'
              ', require_index={}'
              ', pool_size={}'
              ', pool_type={}'
              '{})'.format(
                  var_y,
                  ', {}'.format(var_indices) if has_indices else '',
                  fluid_op,
                  var_x,
                  # attrs
                  has_indices,
                  pool_size,
                  repr(pool_type),
                  name_attr,
              ))
    fluid_op = 'pool{}d'.format(poolnd)
    prog.VarDesc(var_y)
    if has_indices:
        prog.VarDesc(var_indices)
    prog.OpDesc(
        fluid_op,
        ([var_x], 'X'),
        ([var_y] + ([var_indices] if has_indices else []), 'Out', 'Indices'),
        dict(
            global_pooling=False,
            adaptive=True,
            exclusive=True,
            require_index=has_indices,
            pooling_type=pool_type,
            ksize=pool_size,
        ),
    )


def _global_pool(prog, pool_type, inputs, outputs, attrs, value_infos, name=''):
    # I/O
    val_x, = inputs
    val_y, = outputs
    var_x = _make_var_name(val_x)
    var_y = _make_var_name(val_y)

    # interpretation
    input_shape = _shape_or_none(value_infos, val_x)
    output_shape = _shape_or_none(value_infos, val_y)
    assert input_shape is not None or output_shape is not None, 'poolnd not inferred'  # NC...
    if input_shape:
        poolnd = len(input_shape) - 2  # NC...
    elif output_shape:
        poolnd = len(output_shape) - 2  # NC...
    assert 2 <= poolnd <= 3, 'only pool2d and pool3d is supported'

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
        ([var_x], 'X'),
        ([var_y], 'Out'),
        dict(
            global_pooling=True,
            adaptive=False,
            pooling_type=pool_type,
            ksize=[-1, -1],
        ),
    )


def _pool(prog, pool_type, inputs, outputs, attrs, value_infos, name=''):
    # I/O
    val_x, = inputs
    val_y, = outputs[:1]
    var_y = _make_var_name(val_y)

    has_indices = len(outputs) > 1
    if has_indices:
        val_indices = outputs[1]
        var_indices = _make_var_name(val_indices)

    # interpretation
    assert attrs.get(
        'auto_pad',
        'NOTSET') == 'NOTSET', 'only auto_pad = NOTSET is supported'  # optional
    pool_size = attrs['kernel_shape']  # required
    poolnd = len(pool_size)
    assert 2 <= poolnd <= 3, 'only pool2d and pool3d is supported'

    fluid_op = 'pool{}d'.format(poolnd)
    strides = attrs.get('strides', [1] * poolnd)  # optional
    ceil_mode = bool(attrs.get('ceil_mode', 0))  # optional
    pads = attrs.get('pads', [0] * (poolnd * 2))  # optional
    paddings, val_x = _pad_if_asymmetric(prog, pads, val_x, value_infos)
    var_x = _make_var_name(val_x)
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
    if has_indices:
        prog.VarDesc(var_indices)
    prog.OpDesc(
        fluid_op,
        ([var_x], 'X'),
        ([var_y] + ([var_indices] if has_indices else []), 'Out', 'Indices'),
        dict(
            global_pooling=False,
            adaptive=False,
            exclusive=True,
            require_index=has_indices,
            pooling_type=pool_type,
            ksize=pool_size,
            strides=strides,
            paddings=paddings,
            ceil_mode=ceil_mode,
        ),
    )


def _roi_pool(prog, fluid_op, inputs, outputs, attrs, value_infos, name):
    # I/O
    val_x, val_rois = inputs
    val_y, = outputs
    var_x = _make_var_name(val_x)
    var_rois = _make_var_name(val_rois)
    var_y = _make_var_name(val_y)

    # interpretation
    spatial_scale = attrs['spatial_scale']  # required
    pooled_height, pooled_width = attrs['pooled_shape']  # required
    od_attrs = dict(
        pooled_height=pooled_height,
        pooled_width=pooled_width,
        spatial_scale=spatial_scale,
    )
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
                  val_x,
                  var_rois,
                  # attrs
                  spatial_scale,
                  pooled_height,
                  pooled_width,
                  feature_attr,
              ))
    prog.VarDesc(var_y)
    if is_max_pool:
        var_argmax = _make_var_name(name + '.argmax')  # hidden variable
        prog.VarDesc(var_argmax)
    prog.OpDesc(
        fluid_op,
        ([var_x, var_rois], 'X', 'Rois'),
        ([var_y] + ([var_argmax] if is_max_pool else []), 'Out', 'Argmax'),
        od_attrs,
    )


def _interpolate(prog, inputs, outputs, attrs, value_infos, name=''):

    # I/O
    val_x, val_scales = inputs
    val_y, = outputs
    var_x = _make_var_name(val_x)
    var_y = _make_var_name(val_y)

    # interpretation
    # output shape
    out_shape_ = _shape_or_none(value_infos, val_y)
    if out_shape_ is not None:
        assert len(out_shape_) == 4, 'only 4-D Tensor as X and Y supported'
        out_shape_ = out_shape_[2:]
    # try scales
    scales = _const_weight_or_none(value_infos, val_scales)
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
            in_shape = _shape_or_none(value_infos, val_x)
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
        ([var_x], 'X'),
        ([var_y], 'Out'),
        dict(
            interp_method=mode,
            out_h=out_shape_[0],
            out_w=out_shape_[1],
        ),
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
    val_theta, = inputs
    val_grid, = outputs
    var_theta = _make_var_name(val_theta)
    var_grid = _make_var_name(val_grid)

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
        ([var_theta], 'Theta'),
        ([var_grid], 'Output'),
        dict(output_shape=size),  # f**k you API
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
    val_x, val_scale, val_b, val_mean, val_var = inputs
    val_y, = outputs
    var_x = _make_var_name(val_x)
    var_y = _make_var_name(val_y)
    var_saved_mean = name + '.saved_mean'  # dummy output
    var_saved_variance = name + '.saved_variance'  # dummy output

    # interpretation
    fluid_op = 'batch_norm'
    momentum = attrs.get('momentum', .9)  # optional
    epsilon = attrs.get('epsilon', 1e-5)  # optional
    name_attr = ', name={}'.format(repr(name)) if name else ''
    if embed_params:
        assert name != ''
        var_scale = name + '.w_0'
        var_b = name + '.b_0'
        var_mean = name + '.w_1'
        var_var = name + '.w_2'
        value_infos[val_scale].setdefault('embeded_as', []).append(var_scale)
        value_infos[val_b].setdefault('embeded_as', []).append(var_b)
        value_infos[val_mean].setdefault('embeded_as', []).append(var_mean)
        value_infos[val_var].setdefault('embeded_as', []).append(var_var)
        param_attr = ''
    else:
        var_scale = _make_var_name(val_scale)
        var_b = _make_var_name(val_b)
        var_mean = _make_var_name(val_mean)
        var_var = _make_var_name(val_var)
        param_attr = (', param_attr={}, bias_attr={}'
                      ', moving_mean_name={}, moving_variance_name={}').format(
                          repr(var_scale), repr(var_b), repr(var_mean),
                          repr(var_var))

    # generation
    prog.Code('{} = layers.{}({}, is_test=True, data_layout="NCHW"'
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
        ([var_x, var_scale, var_b, var_mean, var_var], 'X', 'Scale', 'Bias',
         'Mean', 'Variance'),
        ([var_y, var_mean, var_saved_mean, var_saved_variance, var_var], 'Y',
         'MeanOut', 'SavedMean', 'SavedVariance', 'VarianceOut'),
        dict(
            is_test=1,
            data_layout='NCHW',
            use_global_stats=False,
            momentum=momentum,
            epsilon=epsilon),
    )


def Cast(prog, inputs, outputs, attrs, value_infos, *args, **kwargs):
    """
    onnx::Cast-9:
    """

    # I/O
    val_input, = inputs
    val_output, = outputs
    var_input = _make_var_name(val_input)
    var_output = _make_var_name(val_output)

    # interpretation
    dtype = attrs['to']  # required
    if not isinstance(dtype, np.dtype):  # additional: possible np.dtype
        dtype = TENSOR_TYPE_TO_NP_TYPE[dtype]
    output_dtype = _dtype_or_none(value_infos, val_output)
    if output_dtype:
        assert dtype == output_dtype, 'dtype of to unmatches output'

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
        ([var_input], 'X'),
        ([var_output], 'Out'),
        dict(
            in_dtype=prog.Dtype(_dtype(value_infos,
                                       val_input)),  # holy, required
            out_dtype=prog.Dtype(dtype),
        ))


def Concat(prog, inputs, outputs, attrs, *args, name='', **kwargs):
    """
    onnx::Concat-4:
    """

    # I/O
    val_concat_result, = outputs
    var_inps = [_make_var_name(val) for val in inputs]
    var_concat_result = _make_var_name(val_concat_result)

    # interpretation
    fluid_op = 'concat'
    axis = attrs['axis']  # required
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('{} = layers.{}({}'
              ', axis={}'
              '{})'.format(
                  var_concat_result,
                  fluid_op,
                  '[' + ', '.join(var_inps) + ']',
                  # attrs
                  axis,
                  name_attr,
              ))
    prog.VarDesc(var_concat_result)
    prog.OpDesc(
        fluid_op,
        (var_inps, *(['X'] * len(var_inps))),
        ([var_concat_result], 'Out'),
        dict(axis=axis),
    )


def Constant(prog, inputs, outputs, attrs, value_infos, *args, **kwargs):
    """
    onnx::Constant-9:
    """

    # I/O
    assert len(inputs) == 0
    val_output, = outputs
    var_output = _make_var_name(val_output)

    # interpretation
    value = attrs['value']  # required
    dtype = np.dtype(value.dtype)
    output_dtype = _dtype_or_none(value_infos, val_output)
    if output_dtype:
        assert dtype == output_dtype, 'tensor dtype unmatches storage dtype'


#    dtype = np.dtype('float32') # HINT: force to float32
    shape = attrs.get('shape', None)  #
    if shape is None:
        shape = _shape_or_none(value_infos, val_output)
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
            ([], ),
            ([var_output], 'Out'),
            dict(
                shape=shape,
                dtype=prog.Dtype(dtype),
                value=value,
            ),
        )
    else:  # list parameter -> const_value
        prog.Code('# {} = {} # passed directly as literal'.format(
            var_output, value))

    value_infos[val_output]['const_value'] = value


def ConstantOfShape(prog, inputs, outputs, attrs, value_infos, *args, **kwargs):
    """
    onnx::ConstantOfShape-9:
    """

    # I/O
    val_shape, = inputs
    val_output, = outputs
    var_shape = _make_var_name(val_shape)

    shape = _const_weight_or_none(value_infos, val_shape)
    if shape is None:
        shape = _shape_or_none(value_infos, val_output)
    assert shape is not None, (
        'given shape is neither const value nor deductible from output, '
        'this is not supported')
    dtype = attrs['value'].dtype
    attrs = attrs.copy()
    attrs.update(dict(shape=shape, dtype=dtype))  # pass const

    prog.Code('# shape:{}={} # const as literal'.format(var_shape, shape))
    prog.Op(
        '',
        'Constant',
        [],
        outputs,  # val
        attrs,
        value_infos,
    )


def Conv(prog,
         inputs,
         outputs,
         attrs,
         value_infos,
         name='',
         embed_params=False,
         *args,
         **kwargs):
    """
    onnx::Conv-1:
    """

    # I/O
    val_x, val_w = inputs[:2]
    val_y, = outputs
    var_y = _make_var_name(val_y)

    has_bias = len(inputs) == 3
    if has_bias:
        val_b, = inputs[2:]

    # interpretation
    assert attrs.get(
        'auto_pad', 'NOTSET'
    ) == 'NOTSET', 'only auto_pad == NOTSET is supported'  # optional
    kernel_shape = _shape(value_infos, val_w)[2:]  # OI...
    assert kernel_shape == attrs[
        'kernel_shape'], 'kernel_shape in attr unmatches value_info'  # HW
    convnd = len(kernel_shape)
    assert 2 <= convnd <= 3, 'only conv2d and conv3d is supported'
    num_out_channels = _shape(value_infos, val_w)[0]  # OI...

    fluid_op = 'conv{}d'.format(convnd)
    num_groups = attrs.get('group', 1)  # optional
    strides = attrs.get('strides', [1] * convnd)  # optional
    dilations = attrs.get('dilations', [1] * convnd)  # optional
    pads = attrs.get('pads', [0] * (convnd * 2))  # optional
    paddings, val_x = _pad_if_asymmetric(prog, pads, val_x, value_infos)
    var_x = _make_var_name(val_x)
    name_attr = ', name={}'.format(repr(name)) if name else ''
    if embed_params:
        assert name != ''
        var_w = name + '.w_0'
        value_infos[val_w].setdefault('embeded_as', []).append(var_w)
        if has_bias:
            var_b = name + '.b_0'
            value_infos[val_b].setdefault('embeded_as', []).append(var_b)
            param_attr = ''
        else:
            param_attr = ', bias_attr=False'
    else:
        var_w = _make_var_name(val_w)
        var_b = _make_var_name(val_b) if has_bias else False
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
    var_conv = name + '.conv'  # hidden variable
    prog.OpDesc(
        fluid_op,
        ([var_x, var_w], 'Input', 'Filter'),  # , 'Bias', 'ResidualData'
        ([var_conv if has_bias else var_y], 'Output'),
        dict(
            strides=strides,
            paddings=paddings,
            dilations=dilations,
            groups=num_groups,
        ))
    if has_bias:
        prog.VarDesc(var_conv)
        prog.IntermediateOp(
            '',
            'Add',
            [var_conv, var_b],  #
            [val_y],
            dict(axis=1),
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
                  name='',
                  embed_params=False,
                  *args,
                  **kwargs):
    """
    onnx::ConvTranspose-1:
    """

    # I/O
    val_x, val_w = inputs[:2]
    val_y, = outputs
    var_y = _make_var_name(val_y)

    has_bias = len(inputs) == 3
    if has_bias:
        val_b, = inputs[2:]

    # interpretation
    assert attrs.get(
        'auto_pad', 'NOTSET'
    ) == 'NOTSET', 'only auto_pad == NOTSET is supported'  # optional
    assert sum(attrs.get(
        'output_padding',
        [])) == 0, 'only zero output_padding is supported'  # optional ?
    kernel_shape = _shape(value_infos, val_w)[2:]  # IO...
    assert kernel_shape == attrs[
        'kernel_shape'], 'kernel_shape in attr unmatches value_info'  # HW
    convnd = len(kernel_shape)
    assert 2 <= convnd <= 3, 'only conv2d_transpose and conv3d_transpose is supported'
    num_out_channels = _shape(value_infos, val_w)[1]  # IO...

    fluid_op = 'conv{}d_transpose'.format(convnd)
    num_groups = attrs.get('group', 1)  # optional
    strides = attrs.get('strides', [1] * convnd)  # optional
    dilations = attrs.get('dilations', [1] * convnd)  # optional
    pads = attrs.get('pads', [0] * (convnd * 2))  # optional
    paddings, val_x = _pad_if_asymmetric(prog, pads, val_x, value_infos)
    var_x = _make_var_name(val_x)
    name_attr = ', name={}'.format(repr(name)) if name else ''
    if embed_params:
        assert name != ''
        var_w = name + '.w_0'
        value_infos[val_w].setdefault('embeded_as', []).append(var_w)
        if has_bias:
            var_b = name + '.b_0'
            value_infos[val_b].setdefault('embeded_as', []).append(var_b)
            param_attr = ''
        else:
            param_attr = ', bias_attr=False'
    else:
        var_w = _make_var_name(val_w)
        var_b = _make_var_name(val_b) if has_bias else False
        param_attr = ', param_attr={}, bias_attr={}'.format(
            repr(var_w),
            repr(var_b) if var_b else False)

    # generation
    prog.Code('{} = layers.{}({}'
              ', num_filters={}'
              #              ', output_size={}'
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
                  kernel_shape,
                  paddings,
                  strides,
                  dilations,
                  num_groups,
                  param_attr,
                  name_attr,
              ))
    var_conv = name + '.conv'  # hidden variable
    prog.OpDesc(
        fluid_op,
        ([var_x, var_w], 'Input', 'Filter'),  # , 'Bias', 'ResidualData'
        ([var_conv if has_bias else var_y], 'Output'),
        dict(
            strides=strides,
            paddings=paddings,
            dilations=dilations,
            # output_size=output_size,
            groups=num_groups,
        ))
    if has_bias:
        prog.VarDesc(var_conv)
        prog.IntermediateOp(
            '',
            'Add',
            [var_conv, var_b],  #
            [val_y],
            dict(axis=1),
            value_infos=value_infos,
            name=(name + '.bias'),
        )
    else:
        prog.VarDesc(var_y)


# should not appear
#def Dropout(
#        prog, inputs, outputs, value_infos,
#        *args, **kwargs):
#    """
#    onnx::Dropout-7:9
#    """
#
#    val_data, = inputs
#    val_output, = outputs[:1]
#
#    _assign(prog,
#            dict(mapping=dict([(val_output, val_data)])),
#            value_infos,
#            )


def Gemm(prog, inputs, outputs, attrs, value_infos, name, *args, **kwargs):
    """
    onnx::Gemm-9:
    """

    # due to fluid fc don't support transposed weight, we use matmul + ew_add
    val_a, val_b, val_c = inputs
    val_y, = outputs

    alpha = attrs.get('alpha', 1.)  # optional
    beta = attrs.get('beta', 1.)  # optional
    trans_a = bool(attrs.get('transA', 0))  # optional
    trans_b = bool(attrs.get('transB', 0))  # optional

    val_mm = name + '_mm'  # explicit variable
    prog.Op(
        '',
        'MatMul',
        [val_a, val_b],
        [val_mm],  # val
        dict(
            transpose_x=trans_a,
            transpose_y=trans_b,
            alpha=alpha,
        ),
        value_infos=value_infos,
        name=val_mm,
    )
    prog.op_descs[-1].attrs.extend(
        prog.OpDescAttrs(dict(
            transpose_X=trans_a,
            transpose_Y=trans_b,
        )))  # f**k you API
    if beta != 0:
        if beta == 1.:  # exactly
            prog.Op(
                '',
                'Add',
                [val_mm, val_c],
                [val_y],  # val
                dict(axis=1),
                value_infos=value_infos,
                name=(name + '_beta'),
            )
        else:
            val_beta = name + '_beta'  # explicit variable
            val_vm = name + '_vm'  # explicit variable
            if beta.is_integer():
                vm_dtype = _dtype_or_none(value_infos, val_c)
                if vm_dtype is None:
                    vm_dtype = np.dtype('float32')
                    _logger.warning(
                        'in %s(%s -> Gemm -> %s): '
                        'attribute "beta" seems to be an interger, '
                        'however dtype can not be inferred, '
                        'still use float32', name, inputs, outputs)
                beta = np.dtype(vm_dtype).type(beta)
            prog.Op(
                '',
                'Constant',
                [],
                [val_beta],  # val
                dict(value=beta),
                value_infos=value_infos,
                name=val_beta,
            )
            prog.Op(
                '',
                'Mul',
                [val_c, val_beta],
                [val_vm],  # val
                dict(),
                value_infos=value_infos,
                name=(name + '_scale'),
            )
            prog.Op(
                '',
                'Add',
                [val_mm, val_vm],
                [val_y],  # val
                dict(axis=1),
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

    return _global_pool(
        prog, 'avg', inputs, outputs, attrs, value_infos, name=name)


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

    return _global_pool(
        prog, 'max', inputs, outputs, attrs, value_infos, name=name)


#def LRN(
#        prog, inputs, outputs, attrs, value_infos, name, # name required
#        *args, **kwargs):
#    """
#    onnx::LRN-1:
#    """
#
#    # I/O
#    val_x, = inputs
#    val_y, = outputs
#    var_x = _make_var_name(val_x)
#    var_y = _make_var_name(val_y)
#
#    # interpretation
#    fluid_op = 'lrn'
#    size = attrs['size'] # required
#    alpha = attrs.get('alpha', 0.0001) # optional
#    beta = attrs.get('beta', 0.75) # optional
#    bias = attrs.get('bias', 1.0) # optional
#    name_attr = ', name={}'.format(repr(name)) if name else ''
#
#    # generation
#    prog.Code('{} = layers.{}({}'
#              ', n={}'
#              ', k={}'
#              ', alpha={}'
#              ', beta={}'
#              '{})'
#              .format(var_y,
#                      fluid_op,
#                      var_x,
#                      # attrs
#                      size,
#                      bias,
#                      alpha,
#                      beta,
#                      name_attr,
#                      ))
#    var_mid = name + '.mid' # hidden variable
#    prog.VarDesc(var_y)
#    prog.VarDesc(var_mid)
#    prog.OpDesc(fluid_op,
#                ([var_x], 'X'),
#                ([var_y, var_mid], 'Out', 'MidOut'),
#                dict(n=size,
#                     k=bias,
#                     alpha=alpha,
#                     beta=beta,
#                     ),
#                )


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
    val_data, = inputs
    val_output, = outputs
    var_data = _make_var_name(val_data)
    var_output = _make_var_name(val_output)

    # interpretation
    pads = attrs['pads']  # required
    mode = attrs.get('mode', 'constant')  # optional
    value = attrs.get('value', 0.)  # optional
    data_shape = _shape_or_none(value_infos, val_data)
    output_shape = _shape_or_none(value_infos, val_output)
    assume_pad2d = False
    if len(pads) == 4:
        assume_pad2d |= mode != 'constant'
        if data_shape:
            assume_pad2d |= data_shape and len(data_shape) == 4  # NCHW
        if output_shape:
            assume_pad2d |= output_shape and len(output_shape) == 4  # NCHW
    od_attrs = dict(pad_value=value)
    if assume_pad2d:
        fluid_op = 'pad2d'
        pad2d_attr = ', mode={}, data_format="NCHW"'.format(repr(mode))
        od_attrs['mode'] = mode
        od_attrs['data_format'] = "NCHW"
    else:
        assert mode == 'constant', 'mode {} is supported only in pad2d'.format(
            mode)
        fluid_op = 'pad'
        pad2d_attr = ''
    paddings = np.array(pads).reshape(
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
        ([var_data], 'X'),
        ([var_output], 'Out'),
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
    val_x, val_slope = inputs
    val_y, = outputs
    var_x = _make_var_name(val_x)
    var_y = _make_var_name(val_y)

    # interpretation
    fluid_op = 'prelu'
    name_attr = ', name={}'.format(repr(name)) if name else ''
    if embed_params:
        assert name != ''
        var_slope = '{}.w_0'.format(val_slope)
        value_infos[val_slope].setdefault('embeded_as', []).append(var_slope)
        param_attr = ''
    else:
        var_slope = _make_var_name(val_slope)
        param_attr = ', param_attr={}'.format(repr(var_slope))

    # generation
    prog.Code('{} = layers.{}({}, mode="all"'
              '{}{})'.format(
                  var_y,
                  fluid_op,
                  var_x,
                  # attrs
                  param_attr,
                  name_attr,
              ))
    prog.VarDesc(var_y)
    prog.OpDesc(
        fluid_op,
        ([var_x], 'X'),
        ([var_y], 'Out'),
        dict(mode='all'),
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
    val_data, val_shape = inputs
    val_reshaped, = outputs
    var_data = _make_var_name(val_data)
    var_shape = _make_var_name(val_shape)
    var_reshaped = _make_var_name(val_reshaped)

    # interpretation
    shape = _const_weight_or_none(value_infos, val_shape)
    is_const_shape = shape and 'const_value' in value_infos[val_shape]
    if shape is None:
        shape = _shape_or_none(value_infos, val_reshaped)


#    assert shape is not None, ('given shape is neither const value nor deductible from output, '
#                               'this is not supported')
    if shape is None:
        shape = [1, -1]  # who knows
        _logger.warning(
            'in %s(%s -> Reshape -> %s): '
            'input "shape" not inferred, use [1, -1] as dummy value, '
            'the behavior of Paddle fluid maybe undefined', name, inputs,
            outputs)
    fluid_op = 'reshape'
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
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
        val_shape_int32 = val_shape + '_int32'  # explicit variable
        var_shape_int32 = _make_var_name(val_shape_int32)
        prog.Op(
            '',
            'Cast',
            [val_shape],
            [val_shape_int32],  # var
            dict(to=np.dtype('int32')),  # use np.dtype
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
    if is_const_shape:
        prog.OpDesc(
            fluid_op,
            ([var_data], 'X'),
            ([var_reshaped, var_xshape], 'Out', 'XShape'),
            dict(shape=shape),
        )
    else:
        prog.OpDesc(
            fluid_op,
            ([var_data, var_shape_int32], 'X', 'Shape'),
            ([var_reshaped, var_xshape], 'Out', 'XShape'),
            dict(shape=shape),
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
    val_data, = inputs
    val_output, = outputs
    var_data = _make_var_name(val_data)
    var_output = _make_var_name(val_output)

    # interpretation
    fluid_op = 'slice'
    axes = attrs['axes']  # required
    starts = attrs['starts']  # required
    ends = attrs['ends']  # required
    shape = _shape_or_none(value_infos, val_data)
    if shape:
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
        ([var_data], 'Input'),
        ([var_output], 'Out'),
        dict(
            axes=axes,
            starts=starts,
            ends=ends,
        ),
    )


def Split(prog, inputs, outputs, attrs, *args, name='', **kwargs):
    """
    onnx::Split-2:
    """

    # I/O
    val_input, = inputs
    var_outs = [_make_var_name(val) for val in outputs]
    var_input = _make_var_name(val_input)

    # interpretation
    fluid_op = 'split'
    split = attrs['split']  # required
    axis = attrs.get('axis', 0)  # optional
    name_attr = ', name={}'.format(repr(name)) if name else ''

    # generation
    prog.Code('{} = layers.{}({}, {}'
              ', dim={}'
              '{})'.format(
                  ', '.join(var_outs),
                  fluid_op,
                  var_input,
                  split,
                  # attrs
                  axis,
                  name_attr,
              ))
    for var_out in var_outs:
        prog.VarDesc(var_out)
    prog.OpDesc(
        fluid_op,
        (var_input, 'X'),
        ([var_outs], *(['Out'] * len(var_outs))),
        dict(
            axis=axis,
            sections=split,
        ),
    )


def Sum(prog, inputs, outputs, *args, **kwargs):
    """
    onnx::Sum-8:
    """

    # I/O
    val_sum, = outputs
    var_inps = [_make_var_name(val) for val in inputs]
    var_sum = _make_var_name(val_sum)

    # interpretation
    fluid_op = 'sums'

    # generation
    prog.Code('{} = layers.{}({})'.format(
        var_sum,
        fluid_op,
        '[' + ', '.join(var_inps) + ']',
        # attrs
    ))
    fluid_op = 'sum'
    prog.VarDesc(var_sum)
    prog.OpDesc(
        fluid_op,
        (var_inps, *(['X'] * len(var_inps))),
        ([var_sum], 'Out'),
        dict(),
    )


def Tile(prog, inputs, outputs, attrs, value_infos, name='', *args, **kwargs):
    """
    onnx::Tile-1:
    """

    # I/O
    val_input, val_repeats = inputs
    val_output, = outputs
    var_input = _make_var_name(val_input)
    var_repeats = _make_var_name(val_repeats)
    var_output = _make_var_name(val_output)

    # interpretation
    repeats = _const_weight_or_none(value_infos, val_repeats)
    assert repeats is not None, 'only const repeats is supported'
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
        ([var_input], 'X'),
        ([var_output], 'Out'),
        dict(expand_times=repeats),
    )


def Transpose(prog, inputs, outputs, attrs, *args, name='', **kwargs):
    """
    onnx::Transpose-1:
    """

    # I/O
    val_data, = inputs
    val_transposed, = outputs
    var_data = _make_var_name(val_data)
    var_transposed = _make_var_name(val_transposed)

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
        ([var_data], 'X'),
        ([var_transposed, var_xshape], 'Out', 'XShape'),
        dict(axis=perm),  # f**k you API
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

    from writer import Program

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
        dict(
            input=dict(shape=(2, 3), dtype=np.float32),
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
    _default(
        prog,
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
    Tile(
        prog, ['input', 'repeats'], ['output'],
        dict(),
        dict(
            repeats=dict(const_value=[1, 2]),
            output=dict(shape=(2, 2, 4), dtype=np.float32)),
        name='Tile')
    logger.info('Tile program:\n%s', prog)
