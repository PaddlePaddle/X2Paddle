#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:50:35 2019

@author: Macrobull
"""

from __future__ import division

import logging, shutil

__all__ = [
    'convert',
]

DEFAULT_ONNX_OPSET_VERSION = 9


def make_var_name(name):
    """
    make a valid variable name in Python code and filename in filesystem
    """

    if name == '':
        return '_'
    if name[0].isdigit():
        return 'var_' + name
    for s in ' \\|/:.-':
        name = name.replace(s, '_')
    if name.startswith('_'):
        name = 'var' + name
    return name


def convert(onnx_model_filename,
            save_dir,
            model_basename='model.py',
            model_func_name='inference',
            embed_params=False,
            onnx_opset_version=None,
            onnx_opset_pedantic=True,
            debug=False,
            **kwargs):
    """
    convert an ONNX model to Paddle fluid Python code and desc pb
    """

    assert isinstance(onnx_model_filename, str)
    assert isinstance(save_dir, str)
    assert isinstance(model_basename, str)
    assert isinstance(model_func_name, str)
    assert onnx_opset_version is None or isinstance(onnx_opset_version, int)

    import onnx

    from onnx.checker import ValidationError
    from onnx.checker import check_model
    from onnx.version_converter import convert_version

    from .onnx_utils import DEFAULT_OP_DOMAIN
    from .onnx_utils import graph_ops, graph_weights
    from .onnx_utils import inferred_model_value_info
    from .onnx_utils import polish_model
    from .writer import Program, Writer

    logger = logging.getLogger('convert')

    # prepare onnx model
    logger.info('loading model: %s ...', onnx_model_filename)
    onnx_model = onnx.load(onnx_model_filename)

    try:
        logger.info('checking model ...')
        check_model(onnx_model)
        if onnx_opset_version is None:  # WORKAROUND: RuntimeError: No Adapter For OP
            logger.warning(
                'opset conversion skipped for onnx_opset_pedantic is OFF')
            logger.info('assumed opset version: %d', DEFAULT_ONNX_OPSET_VERSION)
        else:
            logger.info('using opset version: %d', onnx_opset_version)
            onnx_model = convert_version(onnx_model, onnx_opset_version)
    except ValidationError as e:
        if onnx_opset_pedantic:
            raise e
        else:
            logger.warning('due to onnx_opset_pedantic is OFF')
            logger.warning('the ONNX model sanity checking error is suppressed')
            logger.warning('value_info inferring may be uncompleted')

    # onnx model optimization
    logger.info('model has %d ops', len(onnx_model.graph.node))
    logger.info('optimizing model ...')
    onnx_model = polish_model(onnx_model, checking=onnx_opset_pedantic)

    # prepare filesystem
    shutil.rmtree(save_dir, ignore_errors=True)
    shutil.os.makedirs(save_dir, exist_ok=True)
    logger.info('folder %s cleared', save_dir)

    # DEBUG:
    if debug:
        debug_model_filename, _ = shutil.os.path.splitext(onnx_model_filename)
        onnx.save(onnx_model, debug_model_filename + '.polished.onnx')

    # I/O instances
    onnx_graph = onnx_model.graph
    fluid_program = Program()
    fluid_writer = Writer()

    # model components
    inp_vars = [make_var_name(value.name) for value in onnx_graph.input]
    out_vars = [make_var_name(value.name) for value in onnx_graph.output]
    par_vars = []
    value_infos = inferred_model_value_info(onnx_model)
    value_infos = {
        make_var_name(key): value
        for key, value in value_infos.items()
    }

    # prepare additional value_info
    # for weights
    for name, weight in graph_weights(onnx_graph):
        var_name = make_var_name(name)
        value_info = value_infos[var_name]
        value_info['lod'] = [0]
        value_info['embedded_as'] = []
        value_info['get_weight'] = (lambda w: lambda: w.tolist())(
            weight)  # lazy getter

    logger.info('conversion started')
    # op set conversion
    #    topo = 'backward' if embed_params else 'forward'
    topo = 'forward'
    for name, domain, op_type, inputs, outputs, attrs in graph_ops(onnx_graph,
                                                                   topo=topo):
        op_name = make_var_name(name)
        inputs = list(map(make_var_name, inputs))
        outputs = list(map(make_var_name, outputs))
        logger.debug('translating op %s(%s) %s::%s ...', name, op_name, domain,
                     op_type)
        if domain == DEFAULT_OP_DOMAIN:
            domain = ''
        try:
            fluid_writer.emit_op(
                fluid_program,
                op_name,
                domain,
                op_type,
                inputs,
                outputs,
                attrs,
                value_infos,
                embed_params=embed_params,
            )
        except BaseException as e:
            logger.fatal('conversion failed for:\n\t%s -> %s::%s -> %s', inputs,
                         domain, op_type, outputs)
            raise e
    op_codes = fluid_program.codes
    fluid_program.codes = []
    logger.info('%d ops in, %d ops out', len(onnx_graph.node),
                len(fluid_program.op_descs))

    # type-shape info copy
    for var_name, value_info in value_infos.items():
        fluid_program.VarTypeShapeInfo(var_name, value_info,
                                       remove_batch=False)  #
    bad_vars = []
    for var_name, var_desc in fluid_program.var_descs.items():
        if not var_desc.type.lod_tensor.HasField('tensor'):
            bad_vars.append(var_name)
    if len(bad_vars) > 0:
        logger.warning('type-shape not infered for var %s ...',
                       ', '.join(bad_vars[:5]))
        logger.warning('this causes little problem for PaddlePaddle, '
                       'but Paddle Mobile may not infer correctly')
        logger.warning('please consider running validation with -i '
                       'to invoke type-shape inference in PaddlePaddle')

    # weight writer
    for name, weight in graph_weights(onnx_graph):
        var_name = make_var_name(name)
        par_vars.append(var_name)
        value_info = value_infos[var_name]
        embedded_names = value_info.get('embedded_as', [])
        if embedded_names:
            if len(embedded_names) > 1:
                logger.info(
                    'weight %s is shared between ops, more disk space will be consumed',
                    name)
            logger.debug('saving weight %s(%s[%d], %dB) as %s ...', name,
                         weight.dtype, weight.size, weight.nbytes,
                         embedded_names)
            for embedded_name in embedded_names:  # multiple references
                fluid_writer.write_weight(weight,
                                          shutil.os.path.join(
                                              save_dir, embedded_name),
                                          lod=value_info['lod'])
        else:
            logger.debug('saving weight %s(%s[%d], %dB) to %s ...', name,
                         weight.dtype, weight.size, weight.nbytes, var_name)
            fluid_writer.write_weight(weight,
                                      shutil.os.path.join(save_dir, var_name),
                                      lod=value_info['lod'])
        fluid_writer.emit_param(fluid_program, var_name, value_info)
    param_codes = fluid_program.codes
    fluid_program.codes = []
    logger.info('%d weights converted', len(par_vars))

    # input writer
    external_inputs = []
    for var_name in inp_vars:
        if var_name not in par_vars:
            value_info = value_infos[var_name]
            assert value_info['external']
            external_inputs.append(var_name)
    fluid_writer.emit_inputs(fluid_program,
                             external_inputs,
                             value_infos,
                             remove_batch=False)  # TODO:
    input_codes = fluid_program.codes
    fluid_program.codes = []
    logger.info('%d inputs converted', len(external_inputs))

    # output writer
    external_outputs = []
    for var_name in out_vars:
        if var_name not in par_vars:
            value_info = value_infos[var_name]
            assert value_info['external']
            external_outputs.append(var_name)
    fluid_writer.emit_outputs(fluid_program, external_outputs)
    output_codes = [''] + fluid_program.codes  # add an empty line
    fluid_program.codes = []
    logger.info('%d outputs converted', len(external_outputs))

    # code generation
    header_codes = fluid_writer.header_code(
        model_func_name,
        'From: {}'.format(onnx_model_filename),
    )
    code_filename = shutil.os.path.join(save_dir, model_basename)
    fluid_writer.write_code_file(
        code_filename,
        header_codes,
        input_codes,
        param_codes,
        op_codes,
        output_codes,
    )
    logger.info('code saved to %s, factory function: %s', code_filename,
                model_func_name)

    # desc generation
    desc_filename = shutil.os.path.join(save_dir, '__model__')
    fluid_writer.write_desc_file(
        desc_filename,
        op_descs=fluid_program.op_descs,
        var_descs=list(fluid_program.var_descs.values()),
    )
    logger.info('program saved to %s', desc_filename)

    logger.info('conversion finished')


def main():
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
    basepath, _ = shutil.os.path.splitext(model_filename)
    save_dir = args.output_dir
    save_dir = (save_dir.rstrip(shutil.os.sep)
                if save_dir else basepath) + shutil.os.sep
    embed_params = args.embed_params
    pedantic = args.pedantic
    skip_version_conversion = args.skip_version_conversion

    convert(model_filename,
            save_dir,
            embed_params=embed_params,
            onnx_opset_pedantic=pedantic,
            onnx_skip_version_conversion=skip_version_conversion,
            debug=debug)


if __name__ == '__main__':
    del convert

    from onnx2fluid.conversion import convert

    main()
