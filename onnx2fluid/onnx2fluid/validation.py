#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:17:19 2019

@author: Macrobull
"""

import importlib, logging, os, sys


def flatten_dict(obj, out=None):
    assert isinstance(obj, dict)
    if out is None:
        out = type(obj)()
    for key, value in obj.items():
        if isinstance(value, dict):
            flatten_dict(value, out)
        else:
            assert key not in out
            out[key] = value
    return out


def ensure_list(obj):
    if isinstance(obj, (list, tuple, set)):
        return list(obj)
    return [obj]


def validate(fluid_model_filename,
             golden_data_filename,
             model_func_name='inference',
             atol=1e-3,
             rtol=1e-3,
             save_inference_model=False,
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
        logger.debug('using desc file %s', basename)
        prog, _, var_outs = fluid.io.load_inference_model(fluid_model_dir, exe)
        out_names = var_outs  # HINT: pass var if fetch ops already created
        logger.info('model load passed')
    elif basename.endswith('.py'):  # is Python code
        logger.debug('using code file %s', basename)
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
    logger.info('using golden data %s', golden_data_filename)
    if golden_data_filename.endswith('.npz'):
        test_data = np.load(golden_data_filename, encoding='bytes')
        input_data = test_data['inputs'].tolist()
        output_data = test_data['outputs'].tolist()
    else:
        test_data = np.load(golden_data_filename, encoding='bytes').tolist()
        input_data = test_data['inputs']
        output_data = test_data['outputs']
    input_data = flatten_dict(input_data)
    output_data = flatten_dict(output_data)
    logger.info('found %d I/O golden data, starting test ...',
                len(input_data) + len(output_data))

    # DEBUG: reload test for Python code
    if basename.endswith('.py') and save_inference_model:
        fluid.io.save_inference_model(fluid_model_dir,
                                      input_data.keys(),
                                      var_outs,
                                      exe,
                                      main_program=prog,
                                      export_for_deployment=True)
        logger.info('model re-save passed')
        fluid.io.load_inference_model(fluid_model_dir, exe)
        logger.info('model re-load passed')

    # execute
    outputs = exe.run(prog, feed=input_data, fetch_list=out_names)
    logger.info('execution passed')

    # validate
    passed = True
    for (name, truth), output in zip(output_data.items(), outputs):
        logger.info('testing output {} ...'.format(name))
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
    if passed:
        logger.info('accuracy passed')
    else:
        logger.info('accuracy not passed')

    return passed


if __name__ == '__main__':
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
    args = parser.parse_args()

    logging_format = '[%(levelname)8s]%(name)s::%(funcName)s:%(lineno)04d: %(message)s'
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format=logging_format, level=logging_level)

    debug = args.debug
    fluid_model_filename = args.model[0]
    golden_data_filename = args.test_data
    atol, rtol = args.atol, args.rtol

    validate(fluid_model_filename,
             golden_data_filename,
             atol=atol,
             rtol=rtol,
             save_inference_model=debug)
