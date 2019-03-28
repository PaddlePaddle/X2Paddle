#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:17:19 2019

@author: Macrobull
"""

# import importlib, logging, os, sys
import importlib
import logging
import os
import sys


def _flatten_dict(obj, out=None):
    assert isinstance(obj, dict)
    if out is None:
        out = type(obj)()
    for key, value in obj.items():
        if isinstance(value, dict):
            _flatten_dict(value, out)
        else:
            assert key not in out
            out[key] = value
    return out


def _ensure_list(obj):
    for cls in [list, set, tuple]:
        if isinstance(obj, cls):
            return list(obj)
    return [obj]


def validate(fluid_model_filename,
             golden_data_filename,
             model_func_name='inference',
             precision=1e-4,
             save_inference_model=False):
    """
    inferece the converted Paddle fluid model, validate with given golden data
    """

    import numpy as np
    import paddle.fluid as fluid

    logger = logging.getLogger('validate')

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load model
    fluid_model_dir, basename = os.path.split(fluid_model_filename)
    if basename == '__model__':  # is desc model
        logger.debug('using desc file %s', basename)
        prog, _, var_outs = fluid.io.load_inference_model(fluid_model_dir, exe)
        out_names = var_outs  # HINT: pass var if fetch ops already created
        logger.info('model load passed')
    elif basename.endswith('.py'):  # is python code
        logger.debug('using python code file %s', basename)
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
        var_outs = _ensure_list(var_outs)
        out_names = [var.name for var in var_outs
                     ]  # HINT: pass string to create fetch ops
        logger.info('import passed')

        prog = fluid.default_main_program()
        fluid.io.load_persistables(
            executor=exe, dirname=fluid_model_dir, main_program=prog)
        logger.info('weight load passed')
    else:
        raise ValueError('unsupported Paddle fluid model')

    # load data
    logger.info('using golden data %s', golden_data_filename)
    if golden_data_filename.endswith('.npz'):
        test_data = np.load(golden_data_filename)
        input_data = test_data['inputs'].tolist()
        output_data = test_data['outputs'].tolist()
    else:
        test_data = np.load(golden_data_filename).tolist()
        input_data = test_data['inputs']
        output_data = test_data['outputs']
    input_data = _flatten_dict(input_data)
    output_data = _flatten_dict(output_data)
    logger.info('found %d I/O golden data, starting test ...', len(test_data))

    # DEBUG: reload test for python code
    if basename.endswith('.py') and save_inference_model:
        fluid.io.save_inference_model(
            fluid_model_dir,
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
            np.testing.assert_almost_equal(output, truth, decimal=precision)
        except AssertionError as e:
            passed = False
            logger.error('failed: %s\n', e)
    if passed:
        logger.info('accuracy passed')
    else:
        logger.info('accuracy not passed')


#    globals().update(locals())
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
        '--precision',
        '-p',
        type=int,
        default=4,
        help='assertion decimal for validation',
    )
    args = parser.parse_args()

    logging_format = '[%(levelname)8s]%(name)s::%(funcName)s:%(lineno)04d: %(message)s'
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format=logging_format, level=logging_level)

    debug = args.debug
    fluid_model_filename = args.model[0]
    golden_data_filename = args.test_data
    precision = args.precision

    validate(
        fluid_model_filename,
        golden_data_filename,
        precision=precision,
        save_inference_model=debug)
