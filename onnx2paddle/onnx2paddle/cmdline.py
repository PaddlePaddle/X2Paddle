# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件提供了命令行工具的入口逻辑。

Authors: Macrobull
Date:    2019/02/22 10:25:46
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import logging, shutil, zipfile
import logging
import shutil
import zipfile


__all__ = [
    'main',
]

DEFAULT_ONNX_OPSET_VERSION = 9
DEFAULT_MODEL_MODULE = 'model'
DEFAULT_MODEL_FUNC = 'inference'


def main(**kwargs):
    """主程序入口"""

    try:
        from . import conversion
    except ImportError:
        import conversion

    # imports
    convert = conversion.convert

    logger = logging.getLogger('onnx2paddle')
    debug = kwargs.get('debug', False)

    # prepare arguments
    filename = kwargs['model'][0]
    basepath, _ = shutil.os.path.splitext(filename)
    save_dir = kwargs.get('output_dir', '')
    # model.onnx -> model/
    save_dir = shutil.os.path.dirname(save_dir) if save_dir else basepath
    model_basename = DEFAULT_MODEL_MODULE + '.py'
    model_func_name = DEFAULT_MODEL_FUNC
    embed_params = kwargs.get('embed_params', False)
    onnx_opset_version = DEFAULT_ONNX_OPSET_VERSION
    onnx_opset_pedantic = kwargs.get('pedantic', True)

    # convert
    convert(filename, save_dir,
            model_basename=model_basename,
            model_func_name=model_func_name,
            embed_params=embed_params,
            onnx_opset_version=onnx_opset_version,
            onnx_opset_pedantic=onnx_opset_pedantic,
            debug=debug)

    # validate
    passed = True
    golden_data_filename = kwargs.get('test_data', '')
    if golden_data_filename:
        try:
            from . import validation
        except ImportError:
            import validation

        # imports
        validate = validation.validate

        # in fact fluid can not fully clear the context
        # continuous validation may be inaccurate
        precision = 10 ** -kwargs.get('precision', 4)

        logger.info('starting validation on desc ...')
        passed &= validate(shutil.os.path.join(save_dir, '__model__'),
                           golden_data_filename,
                           precision=precision,
                           )

        logger.info('starting validation on code ...')
        passed &= validate(shutil.os.path.join(save_dir, model_basename),
                           golden_data_filename,
                           model_func_name=model_func_name,
                           precision=precision,
                           save_inference_model=debug, # this overwrite desc file for test
                           )

    if not passed:
        logger.error('validation failed, exit')
        return

    # create zip file
    fn_zip = save_dir.rstrip('/') + '.zip'
    logger.info('compressing file to %s ...', fn_zip)
    fz = zipfile.ZipFile(fn_zip, 'w', compression=zipfile.ZIP_LZMA)
    for fn in shutil.os.listdir(save_dir):
        fz.write(shutil.os.path.join(save_dir, fn), arcname=fn)
    fz.close()
    logger.info('compressing done')


if __name__ == '__main__':
    logging.basicConfig(
            format='[%(levelname)8s]%(name)s::%(funcName)s:%(lineno)04d: %(message)s',
            level=logging.DEBUG,
            )

#    main(model=['../examples/t5.onnx'],
#         output_dir='/tmp/export/',
#         embed_params=False,
#         pedantic=False,
#         test_data='../examples/t5.npz',
#         debug=True)

    main(model=['../examples/shufflenet/model.onnx'],
         output_dir='/tmp/export/',
         embed_params=True,
         pedantic=False,
         test_data='../examples/shufflenet/test_data_set_0.npz',
         debug=True)
