#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:23:09 2019

@author: Macrobull
"""

from __future__ import division

import logging
import numpy as np
import onnx

from collections import OrderedDict as Dict  # as default dict
from onnx.helper import get_attribute_value, make_attribute
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx.numpy_helper import to_array
from onnx.shape_inference import infer_shapes

logger = logging.getLogger(__name__)

__all__ = [
    'print_pb_structure',
    'build_value_refs',
    'node_attrs',
    'node_topo',
    'node_iter',
    'tensor_dtype',
    'tensor_shape',
    'graph_ops',
    'graph_weights',
    'inferred_model_value_info',
    'optimize_model_skip_op_for_inference',
    'optimize_model_strip_initializer',
    'optimize_model_cast',
    'optimize_model_slice',
]

ONNX_INT_MAX = 2**63 - 1

DEFAULT_OP_DOMAIN = 'ai.onnx'


def print_pb_structure(message, loop_iterative=False, depth=0):
    """
    print pb fields in its structure
    """

    if hasattr(message, 'DESCRIPTOR') and hasattr(message.DESCRIPTOR, 'fields'):
        for field in message.DESCRIPTOR.fields:
            print('\t' * depth + '-', field.name)
            print_pb_structure(
                getattr(message, field.name),
                loop_iterative=loop_iterative,
                depth=(depth + 1))

    if loop_iterative and hasattr(message, 'MergeFrom') and hasattr(
            message, '__len__'):
        for idx, item in enumerate(message):
            print('\t' * depth + '-', idx)
            print_pb_structure(
                item, loop_iterative=loop_iterative, depth=(depth + 1))


def build_value_refs(nodes):
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


def get_attribute_value2(attr):
    """
    get_attribute_value with tensor conversion
    """

    if attr.type == onnx.AttributeProto.TENSOR:
        dtype = np.dtype(TENSOR_TYPE_TO_NP_TYPE[attr.t.data_type])
        data = attr.t.raw_data
        value = np.frombuffer(
            data, dtype=dtype, count=(len(data) // dtype.itemsize))
    else:
        value = get_attribute_value(attr)
    return value


def tensor_dtype(tensor):
    """
    get ONNX tensor in np.dtype
    """

    return TENSOR_TYPE_TO_NP_TYPE[tensor.type.tensor_type.elem_type]


def tensor_shape(tensor):
    """
    get ONNX tensor shape
    """

    return [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]


def node_attrs(node):
    """
    convert ONNX node attributes to dict
    """

    return {attr.name: get_attribute_value2(attr)
            for attr in node.attribute}  # dict


def node_topo(nodes, topo='default'):
    """
    build indices with given topology to an ONNX node graph
    """

    if topo == 'default':
        return list(range(len(nodes)))

    node_topo = []
    node_in_degrees = [len(node.input) for node in nodes]
    node_out_degrees = [len(node.output) for node in nodes]
    input_refs, output_refs = build_value_refs(nodes)

    if topo == 'forward':
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
            node_topo.append(node_idx)
            for val_name in nodes[node_idx].output:
                output_refs[val_name].remove(node_idx)
                if len(output_refs[val_name]) > 0:
                    continue
                output_refs.pop(val_name)
                if val_name not in input_refs:
                    continue
                for next_idx in input_refs[val_name]:
                    node_in_degrees[next_idx] -= 1
                    if node_in_degrees[next_idx] == 0:
                        queue.insert(0, next_idx)  # make it lazy
        return node_topo

    if topo == 'backward':
        for val_name in output_refs:
            if val_name not in input_refs:
                for node_idx in output_refs[val_name]:
                    node_out_degrees[node_idx] -= 1
        queue = []
        for node_idx, degree in enumerate(node_out_degrees):
            if degree == 0:
                queue.append(node_idx)
        while len(queue) > 0:
            node_idx = queue.pop(0)
            node_topo.append(node_idx)
            for val_name in nodes[node_idx].input:
                input_refs[val_name].remove(node_idx)
                if len(input_refs[val_name]) > 0:
                    continue
                input_refs.pop(val_name)
                if val_name not in output_refs:
                    continue
                for next_idx in output_refs[val_name]:
                    node_out_degrees[next_idx] -= 1
                    if node_out_degrees[next_idx] == 0:
                        queue.insert(0, next_idx)  # make it lazy
        return node_topo

    raise ValueError('unkown given topo: {}'.format(topo))


def node_iter(nodes, indices=None):
    """
    generator for ONNX node graph with given indices
    """

    if indices is None:
        indices = range(len(nodes))

    for index in indices:
        node = nodes[index]
        name = node.name
        domain = node.domain
        op_type = node.op_type
        inputs = list(node.input)
        outputs = list(node.output)
        attrs = node_attrs(node)

        if name == '':
            name = 'op_' + str(index)
        if domain == '':
            domain = DEFAULT_OP_DOMAIN

        yield name, domain, op_type, inputs, outputs, attrs


def graph_ops(graph, topo='default'):
    """
    generator for ONNX node graph with given topology
    """

    if not isinstance(graph, onnx.GraphProto):
        logger.error('graph is not a GraphProto instance')
        return

    return node_iter(graph.node, node_topo(graph.node, topo))


def graph_weights(graph):
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


def inferred_model_value_info(model):
    """
    collect value/type info for an ONNX model
    """

    model = infer_shapes(model)
    graph = model.graph
    value_info = Dict()
    for item in graph.value_info:
        value_info[item.name] = dict(
            dtype=tensor_dtype(item),
            shape=tensor_shape(item),
            external=False,
        )
    for item in graph.input:
        assert item.name not in value_info
        value_info[item.name] = dict(
            dtype=tensor_dtype(item),
            shape=tensor_shape(item),
            external=True,
        )
    for item in graph.output:
        #        assert item.name not in value_info, 'bypass-model not supported'
        value_info[item.name] = dict(
            dtype=tensor_dtype(item),
            shape=tensor_shape(item),
            external=True,
        )
    return value_info


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


def optimize_model_skip_op_for_inference(model, op_list=None):
    """
    skip ops can be bypassed for inference
    """
    if op_list is None:
        op_list = ['Dropout']

    nodes = model.graph.node
    input_refs, output_refs = build_value_refs(nodes)

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
            logger.warning(
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
            logger.debug('skip op %d: %s -> %s -> %s', node_idx, input_name,
                         node.op_type, output_name)
        elif processed == 0:
            logger.warning('weird, no node processed')
        else:
            logger.warning('standalone op %d: %s -> %s -> %s not skipped',
                           node_idx, input_name, node.op_type, output_name)

    nodes_to_remove.sort(reverse=True)
    for node_idx in nodes_to_remove:
        ret_nodes.pop(node_idx)

    return ret


def optimize_model_strip_initializer(model, keep_input_only=True):
    """
    strip weights for inference
    """

    nodes = model.graph.node
    input_refs, output_refs = build_value_refs(nodes)
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
            logger.debug('initializer %s(%s[%d]) stripped', name, dtype,
                         len(initializer.raw_data) // dtype.itemsize)

    # strip inputs
    ret.graph.ClearField('input')
    ret_inputs = ret.graph.input
    for item in model.graph.input:
        name = item.name
        if name in input_refs or name in out_names:
            ret_inputs.add().CopyFrom(item)
        else:
            logger.debug('input %s(%s%s) stripped', name, tensor_dtype(item),
                         tensor_shape(item))
    return ret


def optimize_model_cast(model):
    """
    strip cascade and unecessary onnx::Cast
    """

    nodes = model.graph.node
    input_refs, output_refs = build_value_refs(nodes)
    value_info = inferred_model_value_info(model)

    ret = type(model)()
    ret.CopyFrom(model)
    ret.graph.ClearField(
        'value_info')  # WORKAROUND: onnx do not drop old value_info
    ret_nodes = ret.graph.node
    nodes_to_remove = []
    for node_idx, node in enumerate(nodes):
        if not (node.domain == DEFAULT_OP_DOMAIN or node.domain == ''):
            continue
        if not (node.op_type == 'Cast'):
            continue
        attrs = node_attrs(node)
        output_dtype = TENSOR_TYPE_TO_NP_TYPE[attrs['to']]
        input_name = node.input[0]
        info = value_info.get('input_name', None)  # relax for un-inferrable
        if info is None:
            continue
        input_dtype = info.get('dtype', None)
        if input_dtype is None or input_dtype != output_dtype:
            continue

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
            logger.debug('skip %s: %s -> %s Cast op', node.name, input_dtype,
                         output_dtype)
        elif processed == 0:
            logger.warning('weird, no node processed')
        else:
            logger.debug('keep standalone %s: %s -> %s Cast op', node.name,
                         input_dtype, output_dtype)

    nodes_to_remove.sort(reverse=True)
    for node_idx in nodes_to_remove:
        ret_nodes.pop(node_idx)

    return ret


def optimize_model_slice(model):
    """
    strip cascade and unecessary onnx::Slice
    """

    nodes = model.graph.node
    input_refs, output_refs = build_value_refs(nodes)

    def _build_slice_node_chain(node_idx):
        chain = []
        while True:
            node = nodes[node_idx]
            if not (node.domain == DEFAULT_OP_DOMAIN or node.domain == ''):
                return chain
            if not node.op_type == 'Slice':
                return chain
            chain.append(node_idx)
            output_name = node.output[0]
            if output_name not in input_refs or len(
                    input_refs[output_name]) != 1:
                return chain
            node_idx = list(input_refs[output_name])[0]

    # axis: (start, end)
    def _merge_slice(slice_chain):
        merged_slice = dict()
        for slice_node_idx in slice_chain:
            node = nodes[slice_node_idx]
            attrs = node_attrs(node)
            for axis, start, end in zip(attrs['axes'], attrs['starts'],
                                        attrs['ends']):
                if start == 0 and end == ONNX_INT_MAX:
                    continue
                if axis in merged_slice:
                    prev_start, prev_end = merged_slice[axis]
                    start += prev_start if start >= 0 else 0 if prev_end == ONNX_INT_MAX else prev_end
                    end += prev_start if end >= 0 else 0 if prev_end == ONNX_INT_MAX else prev_end
                merged_slice[axis] = (start, end)
        return merged_slice

    ret = type(model)()
    ret.CopyFrom(model)
    ret.graph.ClearField(
        'value_info')  # WORKAROUND: onnx do not drop old value_info
    ret_nodes = ret.graph.node
    nodes_to_remove = []
    for node_idx in range(len(nodes)):
        slice_chain = _build_slice_node_chain(node_idx)
        if len(slice_chain) == 0:
            continue
        merged_slice = _merge_slice(slice_chain)
        if len(merged_slice) > 0 and len(slice_chain) == 1:  # no need to merge
            continue

        attrs = dict(axes=[], starts=[], ends=[])
        for axis, (start, end) in merged_slice.items():
            attrs['axes'].append(axis)
            attrs['starts'].append(start)
            attrs['ends'].append(end)
        first_node = nodes[slice_chain[0]]
        last_node = nodes[slice_chain[-1]]
        input_name = first_node.input[0]
        output_name = last_node.output[0]
        processed = -1
        if output_name in input_refs:  # 0, [1...]
            new_input_name = first_node.output[0] if len(
                merged_slice) > 0 else input_name
            processed = skip_node_forward(ret_nodes, output_name,
                                          new_input_name, input_refs)
            if processed > 0:
                if len(merged_slice) > 0:
                    remain_idx = slice_chain[0]
                    remove_chain = slice_chain[1:]
                    slice_node = ret_nodes[remain_idx]
                    for attr in slice_node.attribute:
                        attr.CopyFrom(
                            make_attribute(attr.name, attrs[attr.name]))
                    logger.debug('merged slice chain %s -> %s%s -> %s',
                                 input_name, remain_idx, remove_chain,
                                 output_name)
                else:
                    remove_chain = slice_chain

        if processed < 0 and input_name in output_refs:
            new_output_name = last_node.input[0] if len(
                merged_slice) > 0 else output_name
            processed = skip_node_backward(ret_nodes, input_name,
                                           new_output_name, output_refs)
            if processed > 0:
                if len(merged_slice) > 0:
                    remain_idx = slice_chain[-1]
                    remove_chain = slice_chain[:-1]
                    slice_node = ret_nodes[remain_idx]
                    for attr in slice_node.attribute:
                        attr.CopyFrom(
                            make_attribute(attr.name, attrs[attr.name]))
                    logger.debug('merged slice chain %s -> %s%s -> %s',
                                 input_name, remove_chain, remain_idx,
                                 output_name)
                else:
                    remove_chain = slice_chain

        if processed > 0:
            nodes_to_remove.extend(remove_chain)
            if len(merged_slice) == 0:
                logger.debug('skip slice chain %s -> %s -> %s', input_name,
                             slice_chain, output_name)
        elif processed < 0:  # NEVERFIX: not merge standalone slice chain
            logger.debug('keep standalone slice chain %s -> %s -> %s',
                         input_name, slice_chain, output_name)

    nodes_to_remove.sort(reverse=True)
    for node_idx in nodes_to_remove:
        ret_nodes.pop(node_idx)

    return ret


if __name__ == '__main__':
    logging.basicConfig(
        format=
        '[%(levelname)8s]%(name)s::%(funcName)s:%(lineno)04d: %(message)s',
        level=logging.DEBUG,
    )

    from onnx.checker import check_model
    from onnx.utils import polish_model
    from onnx.version_converter import convert_version

    model = onnx.load('../examples/t1.onnx')
    print_pb_structure(model, loop_iterative=False)

    check_model(model)
    model = convert_version(model, 9)
    model = optimize_model_skip_op_for_inference(model)
    model = optimize_model_strip_initializer(model)
    model = optimize_model_cast(model)
    model = optimize_model_slice(model)
    model = polish_model(model)

    onnx.save(model, '/tmp/optimized.onnx')

    graph = model.graph
    value_info = inferred_model_value_info(model)

    name = graph.name
    inputs = [value.name for value in graph.input]
    outputs = [value.name for value in graph.output]
    weights = []

    logger.info('ops:')
    for name, domain, op_type, _, _, attrs in graph_ops(graph, topo='forward'):
        logger.info('%s %s::%s: %s', name, domain, op_type, attrs)

    logger.info('weights:')
    for name, array in graph_weights(graph):
        weights.append(name)
        logger.info('%s: %s', name, array.shape)

    logger.info('inputs:')
    external_inputs = []
    for name in inputs:
        if name not in weights:
            external_inputs.append(name)
            logger.info('%s: %s', name, value_info[name]['shape'])

    logger.info('outputs:')
    external_outputs = []
    for name in outputs:
        if name not in weights:
            external_outputs.append(name)
            logger.info('%s: %s', name, value_info[name]['shape'])
