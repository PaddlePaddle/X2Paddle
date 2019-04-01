#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:50:35 2019

@author: Macrobull
"""

from __future__ import division

import logging, shutil
#import logging
#import shutil

__all__ = [
    'convert',
]


def convert(onnx_model_filename,
            save_dir,
            model_basename='model.py',
            model_func_name='inference',
            embed_params=False,
            onnx_opset_version=9,
            onnx_opset_pedantic=True,
            onnx_skip_version_conversion=False,
            debug=False):
    """
    convert an ONNX model to Paddle fluid Python code and desc pb
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
        if onnx_skip_version_conversion:  # WORKAROUND: RuntimeError: No Adapter For OP
            logger.debug('assumed opset version: %d', onnx_opset_version)
            logger.warning(
                'opset conversion skipped for onnx_opset_pedantic is OFF')
        else:
            logger.debug('using opset version: %d', onnx_opset_version)
            onnx_model = convert_version(onnx_model, onnx_opset_version)
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
    fluid_program = Program()
    fluid_writer = Writer()

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
        value_info['get_weight'] = (lambda w: lambda: w.tolist())(
            weight)  # lazy getter

    logger.info('conversion started')
    # op set conversion
    #    topo = 'backward' if embed_params else 'forward'
    topo = 'forward'
    for name, domain, op_type, inputs, outputs, attrs in graph_ops(
            onnx_graph, topo=topo):
        logger.debug('translating op %s %s::%s ...', name, domain, op_type)
        if domain == DEFAULT_OP_DOMAIN:
            domain = ''
        try:
            fluid_writer.emit_op(
                fluid_program,
                name,
                domain,
                op_type,
                inputs,
                outputs,
                attrs,
                graph_value_infos,
                embed_params=embed_params,
            )
        except BaseException as e:
            logger.fatal('conversion failed for:\n\t%s -> %s::%s -> %s', inputs,
                         domain, op_type, outputs)
            raise e
    op_codes = fluid_program.codes
    fluid_program.codes = []
    logger.info('%d ops converted', len(fluid_program.op_descs))

    # weight writer
    for name, weight in graph_weights(onnx_graph):
        graph_params.append(name)
        value_info = graph_value_infos[name]
        var_names = value_info.get('embeded_as', [])
        if var_names:
            if len(var_names) > 1:
                logger.info(
                    'weight %s is shared between ops, more disk space will be consumed',
                    name)
            logger.debug('saving weight %s(%s[%d], %dB) as %s ...', name,
                         weight.dtype, weight.size, weight.nbytes, var_names)
            for var_name in var_names:  # multiple references
                fluid_writer.write_weight(
                    weight, shutil.os.path.join(save_dir, var_name))
        else:
            logger.debug('saving weight %s(%s[%d], %dB) to %s ...', name,
                         weight.dtype, weight.size, weight.nbytes,
                         make_var_name(name))
            fluid_writer.write_weight(
                weight, shutil.os.path.join(save_dir, make_var_name(name)))
        fluid_writer.emit_param(fluid_program, name, value_info)
    param_codes = fluid_program.codes
    fluid_program.codes = []
    logger.info('%d weights converted', len(graph_params))

    # input writer
    external_inputs = []
    for name in graph_inputs:
        if name not in graph_params:
            value_info = graph_value_infos[name]
            assert value_info['external']
            external_inputs.append(name)
    fluid_writer.emit_inputs(
        fluid_program, external_inputs, graph_value_infos,
        remove_batch=False)  # TODO:
    input_codes = fluid_program.codes
    fluid_program.codes = []
    logger.info('%d inputs converted', len(external_inputs))

    # output writer
    external_outputs = []
    for name in graph_outputs:
        if name not in graph_params:
            value_info = graph_value_infos[name]
            assert value_info['external']
            external_outputs.append(name)
    fluid_writer.emit_outputs(fluid_program, external_outputs)
    output_codes = [''] + fluid_program.codes  # add an empty line
    fluid_program.codes = []
    logger.info('%d outputs converted', len(external_outputs))

    # code generation
    header_codes = fluid_writer.header_code(
        model_func_name, 'From: {}'.format(onnx_model_filename))
    code_filename = shutil.os.path.join(save_dir, model_basename)
    fluid_writer.write_code_file(code_filename, header_codes, input_codes,
                                 param_codes, op_codes, output_codes)
    logger.info('code saved to %s, factory function: %s', code_filename,
                model_func_name)

    # desc generation
    desc_filename = shutil.os.path.join(save_dir, '__model__')
    fluid_writer.write_desc_file(
        desc_filename,
        op_descs=fluid_program.op_descs,
        var_descs=fluid_program.var_descs,
    )
    logger.info('program saved to %s', desc_filename)

    logger.info('conversion finished')


#    globals().update(locals())

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='onnx2fluid.convert',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'model',
        nargs=1,
        help='path to model.onnx',
    )
    parser.add_argument(
        '--debug',
        '-d',
        action='store_true',
        help='enable debug logging and checking',
    )
    parser.add_argument(
        '--output_dir',
        '-o',
        type=str,
        default='',
        help='output directory',
    )
    parser.add_argument(
        '--embed_params',
        '-e',
        action='store_true',
        help='try to embed parameters for trainable Paddle fluid layers',
    )
    parser.add_argument(
        '--pedantic',
        action='store_true',
        default=True,
        help='accept and convert only standard ONNX opset',
    )
    parser.add_argument(
        '--no-pedantic',
        '-x',
        action='store_false',
        dest='pedantic',
        help='process non-standard ONNX ops, this may lead to fails',
    )
    parser.add_argument(
        '--skip-version-conversion',
        '-y',
        action='store_true',
        default=False,
        help='skip ONNX op version conversion, workaround for RumtimeErrors',
    )
    args = parser.parse_args()

    logging_format = '[%(levelname)8s]%(name)s::%(funcName)s:%(lineno)04d: %(message)s'
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format=logging_format, level=logging_level)

    debug = args.debug
    model_filename = args.model[0]
    save_dir = args.output_dir
    embed_params = args.embed_params
    pedantic = args.pedantic
    skip_version_conversion = args.skip_version_conversion

    convert(
        model_filename,
        save_dir,
        embed_params=embed_params,
        onnx_opset_pedantic=pedantic,
        onnx_skip_version_conversion=skip_version_conversion,
        debug=debug)
