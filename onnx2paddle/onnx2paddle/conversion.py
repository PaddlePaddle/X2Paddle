#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:50:35 2019

@author: Macrobull
"""

from __future__ import division

# import logging, shutil
import logging
import shutil


__all__ = [
    'convert',
]


def convert(onnx_model_filename, save_dir,
            model_basename='model.py', model_func_name='inference',
            embed_params=False,
            onnx_opset_version=9, onnx_opset_pedantic=True,
            debug=False):
    """
    convert an ONNX model to Paddle Python code and desc pb
    """

    import onnx

    from onnx.checker import ValidationError
    from onnx.checker import check_model
    from onnx.utils import polish_model
    from onnx.version_converter import convert_version

    try:
        from . import onnx_utils, writer
    except ImportError:
        import onnx_utils, writer

    # imports
    DEFAULT_OP_DOMAIN = onnx_utils.DEFAULT_OP_DOMAIN
    graph_ops, graph_weights = onnx_utils.graph_ops, onnx_utils.graph_weights
    inferred_model_value_info = onnx_utils.inferred_model_value_info
    optimize_model_skip_op_for_inference = onnx_utils.optimize_model_skip_op_for_inference
    optimize_model_strip_initializer = onnx_utils.optimize_model_strip_initializer
    optimize_model_cast = onnx_utils.optimize_model_cast
    optimize_model_slice = onnx_utils.optimize_model_slice
    Program, Writer = writer.Program, writer.Writer
    make_var_name = writer.make_var_name

    logger = logging.getLogger('convert')

    # prepare onnx model
    logger.info('loading model: %s ...', onnx_model_filename)
    onnx_model = onnx.load(onnx_model_filename)
    try:
        logger.info('checking model ...')
        check_model(onnx_model)
        logger.debug('using opset version: %d', onnx_opset_version)
        if onnx_opset_pedantic: # WORKAROUND: RuntimeError: No Adapter For OP
            onnx_model = convert_version(onnx_model, onnx_opset_version)
        else: # TODO: add new argument for this option
            logger.warning('opset conversion skipped for onnx_opset_pedantic is OFF')
        onnx_model = polish_model(onnx_model)
    except ValidationError as e:
        if onnx_opset_pedantic:
            raise e
        else:
            logger.warning('due to onnx_opset_pedantic is OFF')
            logger.warning('the ONNX model sanity checking error is suppressed')
            logger.warning('value_info inferring may be uncompleted')
    # onnx model optimization
    logger.info('optimizing model ...')
    onnx_model = optimize_model_skip_op_for_inference(onnx_model)
    onnx_model = optimize_model_strip_initializer(onnx_model)
    onnx_model = optimize_model_cast(onnx_model)
    onnx_model = optimize_model_slice(onnx_model)

    # prepare filesystem
    shutil.rmtree(save_dir, ignore_errors=True)
    shutil.os.makedirs(save_dir, exist_ok=True)
    logger.info('folder %s cleared', save_dir)

    # DEBUG:
    if debug:
        model = onnx.shape_inference.infer_shapes(onnx_model)
        debug_model_filename, _ = shutil.os.path.splitext(onnx_model_filename)
        onnx.save(model, debug_model_filename + '.optimized_and_inffered.onnx')
#        onnx.save(model, '/tmp/export/optimized_and_inffered.onnx')

    # I/O instances
    onnx_graph = onnx_model.graph
    paddle_program = Program()
    paddle_writer = Writer()

    # model components
#    graph_name = onnx_graph.name
    graph_inputs = [value.name for value in onnx_graph.input]
    graph_outputs = [value.name for value in onnx_graph.output]
    graph_params = []
    graph_value_infos = inferred_model_value_info(onnx_model)

    # prepare additional value_info
    # for weights
    for name, weight in graph_weights(onnx_graph):
        value_info = graph_value_infos[name]
        value_info['embeded_as'] = []
        value_info['get_weight'] = lambda: weight.tolist() # lazy getter

    logger.info('conversion started')
    # op set conversion
#    topo = 'backward' if embed_params else 'forward'
    topo = 'forward'
    for name, domain, op_type, inputs, outputs, attrs in graph_ops(onnx_graph, topo=topo):
        logger.debug('translating op %s %s::%s ...', name, domain, op_type)
        if domain == DEFAULT_OP_DOMAIN:
            domain = ''
        try:
            paddle_writer.emit_op(paddle_program, name, domain, op_type,
                                  inputs, outputs, attrs,
                                  graph_value_infos,
                                  embed_params=embed_params,
                                  )
        except BaseException as e:
            logger.fatal('conversion failed for:\n\t%s -> %s::%s -> %s',
                         inputs, domain, op_type, outputs)
            raise e
    op_codes = paddle_program.codes
    paddle_program.codes = []
    logger.info('%d ops converted', len(paddle_program.op_descs))

    # weight writer
    for name, weight in graph_weights(onnx_graph):
        graph_params.append(name)
        value_info = graph_value_infos[name]
        var_names = value_info.get('embeded_as', [])
        if var_names:
            if len(var_names) > 1:
                logger.info('weight %s is shared between ops, more disk space will be consumed', name)
            logger.debug('saving weight %s with size of %d, in %d bytes, as %s ...',
                         name, weight.size, weight.nbytes, var_names)
            for var_name in var_names: # multiple references
                paddle_writer.write_weight(weight, shutil.os.path.join(save_dir, var_name))
        else:
            logger.debug('saving weight %s with size of %d, in %d bytes, to %s ...',
                         name, weight.size, weight.nbytes, make_var_name(name))
            paddle_writer.write_weight(weight, shutil.os.path.join(save_dir, make_var_name(name)))
        paddle_writer.emit_param(paddle_program, name, value_info)
    param_codes = paddle_program.codes
    paddle_program.codes = []
    logger.info('%d weights converted', len(graph_params))

    # input writer
    external_inputs = []
    for name in graph_inputs:
        if name not in graph_params:
            value_info = graph_value_infos[name]
            assert value_info['external']
            external_inputs.append(name)
    paddle_writer.emit_inputs(paddle_program, external_inputs, graph_value_infos, remove_batch=False) # TODO:
    input_codes = paddle_program.codes
    paddle_program.codes = []
    logger.info('%d inputs converted', len(external_inputs))

    # output writer
    external_outputs = []
    for name in graph_outputs:
        if name not in graph_params:
            value_info = graph_value_infos[name]
            assert value_info['external']
            external_outputs.append(name)
    paddle_writer.emit_outputs(paddle_program, external_outputs)
    output_codes = [''] + paddle_program.codes # add an empty line
    paddle_program.codes = []
    logger.info('%d outputs converted', len(external_outputs))

    # code generation
    code_filename = shutil.os.path.join(save_dir, model_basename)
    paddle_writer.write_code_file(code_filename, paddle_writer.header_code(model_func_name),
                                  input_codes, param_codes, op_codes, output_codes)
    logger.info('code saved to %s, factory function: %s', code_filename, model_func_name)

    # desc generation
    desc_filename = shutil.os.path.join(save_dir, '__model__')
    paddle_writer.write_desc_file(desc_filename,
                                  op_descs=paddle_program.op_descs,
                                  var_descs=paddle_program.var_descs,
                                  )
    logger.info('program saved to %s', desc_filename)

    logger.info('conversion finished')
#    globals().update(locals())


if __name__ == '__main__':
    logging.basicConfig(
            format='[%(levelname)8s]%(name)s::%(funcName)s:%(lineno)04d: %(message)s',
            level=logging.DEBUG,
            )

    model_list = [
            '../examples/t1.onnx',
            '../examples/t2.onnx',
            '../examples/t3.onnx',
            '../examples/t4.onnx',
            '../examples/t5.onnx',
            '../examples/t6.onnx',
#            '../examples/t7.onnx',
#            '../examples/t8.onnx',
    ]

    for model in model_list:
        pathname, _ = shutil.os.path.splitext(model)
        convert(model, pathname,
                onnx_opset_pedantic=False, debug=True)
        convert(model, pathname + '.embeded',
                embed_params=True, onnx_opset_pedantic=False, debug=True)
