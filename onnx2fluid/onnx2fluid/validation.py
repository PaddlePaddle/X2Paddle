#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:17:19 2019

@author: Macrobull
"""

import importlib, logging, os, sys

logger = logging.getLogger(__name__)

__all__ = [
    'fluid_prog_shape_infer',
    'validate',
]


def flatten_dict(obj, out=None):
    assert isinstance(obj, dict), 'dict type required'

    if out is None:
        out = type(obj)()
    for key, value in obj.items():
        if isinstance(value, dict):
            flatten_dict(value, out)
        else:
            assert key not in out, 'key conflicted'
            out[key] = value
    return out


def ensure_list(obj):
    if isinstance(obj, (list, tuple, set)):
        return list(obj)
    return [obj]


def fluid_prog_shape_infer(prog):
    """
    additional type-shape inference for fluid program
    """

    import paddle.fluid as fluid

    assert isinstance(prog, fluid.framework.Program)

    logger.info('performing type-shape inference ...')
    for block in prog.blocks:
        block_desc = block.desc

        for idx_op in range(block_desc.op_size()):
            op_desc = block_desc.op(idx_op)
            if op_desc.type() in ('feed', 'fetch'):
                continue

            op_desc.infer_var_type(block_desc)
            op_desc.infer_shape(block_desc)

        for var_name, var in block.vars.items():
            var_desc = var.desc
            if var_desc.type() != fluid.core.VarDesc.VarType.LOD_TENSOR:
                continue

            # WORKAROUND: dirty way to give dtype to partial-infered vars
            # which could not be cleared!
            try:
                var.to_string(True)
            except ValueError:
                var_desc.set_dtype(fluid.core.VarDesc.VarType.FP32)
                logger.debug('dtype of var %s not inferred, float32 assumed',
                             var_name)


def validate(fluid_model_filename,
             golden_data_filename='',
             atol=1e-3,
             rtol=1e-3,
             model_func_name='inference',
             save_inference_model=False,
             inference_input_names=None,
             **kwargs):
    """
    inference the converted Paddle fluid model, validate with given golden data
    """

    import numpy as np
    import paddle.fluid as fluid

    logger = logging.getLogger('validate')

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load model
    fluid_model_dir, basename = os.path.split(fluid_model_filename)
    if basename == '__model__':  # is desc program
        logger.info('using desc file %s', basename)
        prog, _, var_outs = fluid.io.load_inference_model(fluid_model_dir, exe)
        out_names = var_outs  # HINT: pass var if fetch ops already created
        logger.info('model load passed')
    elif basename.endswith('.py'):  # is Python code
        logger.info('using code file %s', basename)
        module_name, _ = os.path.splitext(basename)
        sys_path = sys.path.copy()
        sys.path.append(fluid_model_dir)
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, model_func_name)
        except AttributeError:
            module_name = module_name + '.' + module_name
            module = importlib.import_module(module_name)
            func = getattr(module, model_func_name)
        sys.path = sys_path
        logger.debug('from %s imported %s: %s', module_name, model_func_name,
                     func)

        var_outs = func()
        var_outs = ensure_list(var_outs)
        out_names = [var.name for var in var_outs
                     ]  # HINT: pass string to create fetch ops
        logger.info('import passed')

        prog = fluid.default_main_program()
        fluid.io.load_persistables(executor=exe,
                                   dirname=fluid_model_dir,
                                   main_program=prog)
        logger.info('weight load passed')
    else:
        raise ValueError('unsupported Paddle fluid model filename')

    # load data
    if golden_data_filename:
        logger.info('using golden data %s', golden_data_filename)
        if golden_data_filename.endswith('.npz'):
            test_data = np.load(
                golden_data_filename,
                encoding='bytes',
                allow_pickle=True,
            )
            input_data = test_data['inputs'].tolist()
            output_data = test_data['outputs'].tolist()
        else:
            test_data = np.load(
                golden_data_filename,
                encoding='bytes',
                allow_pickle=True,
            ).tolist()
            input_data = test_data['inputs']
            output_data = test_data['outputs']

        input_data = flatten_dict(input_data)
        output_data = flatten_dict(output_data)
        input_names = input_data.keys()
        output_names = output_data.keys()
        logger.info('with %d inputs and %d outputs', len(input_data),
                    len(output_data))
    else:
        assert inference_input_names, 'input names required for type-shape inference'

        input_names = inference_input_names
        logger.info('using input names: %s', ', '.join(input_names))

    # type-shape inference and re-save
    if save_inference_model:
        fluid_prog_shape_infer(prog)
        fluid.io.save_inference_model(fluid_model_dir,
                                      input_names,
                                      var_outs,
                                      exe,
                                      main_program=prog,
                                      export_for_deployment=True)
        logger.info('model re-save passed')
        fluid.io.load_inference_model(fluid_model_dir, exe)
        logger.info('model re-load passed')

    if not golden_data_filename:
        return True

    # execute
    outputs = exe.run(prog, feed=input_data,
                      fetch_list=out_names)  # out_names can be vars
    logger.info('execution passed')

    # validate
    passed = True
    for (name, truth), output in zip(output_data.items(), outputs):
        logger.info('testing on output {} ...'.format(name))
        try:
            np.testing.assert_allclose(output,
                                       truth,
                                       rtol=rtol,
                                       atol=atol,
                                       equal_nan=False,
                                       verbose=True)
        except AssertionError as e:
            passed = False
            logger.error('failed: %s\n', e)
    logger.info('accuracy %spassed', '' if passed else 'not ')
    return passed


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='onnx2fluid.validate',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'model',
        nargs=1,
        help='path to model.py or __model__',
    )
    parser.add_argument(
        '--debug',
        '-d',
        action='store_true',
        help='enable debug logging and checking',
    )
    parser.add_argument(
        '--test_data',
        '-t',
        type=str,
        default='',
        help='I/O golden data for validation, e.g. test.npy, test.npz',
    )
    parser.add_argument(
        '--atol',
        '-p',
        type=float,
        default=1e-3,
        help='assertion absolute tolerance for validation',
    )
    parser.add_argument(
        '--rtol',
        type=float,
        default=1e-2,
        help='assertion relative tolerance for validation',
    )
    parser.add_argument(
        '--infer_inputs',
        '-i',
        nargs='?',
        default=None,
        const='',
        help=
        'perform type-shape inference with given input names and re-save model',
    )
    args = parser.parse_args()

    logging_format = '[%(levelname)8s]%(name)s::%(funcName)s:%(lineno)04d: %(message)s'
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format=logging_format, level=logging_level)

    #	debug = args.debug
    fluid_model_filename = args.model[0]
    golden_data_filename = args.test_data
    atol, rtol = args.atol, args.rtol
    save_inference_model = args.infer_inputs is not None
    inference_input_names = args.infer_inputs.split(
        ',') if args.infer_inputs else None

    validate(fluid_model_filename,
             golden_data_filename=golden_data_filename,
             atol=atol,
             rtol=rtol,
             save_inference_model=save_inference_model,
             inference_input_names=inference_input_names)


if __name__ == '__main__':
    main()
