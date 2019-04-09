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

import logging, shutil, zipfile
#import logging
#import shutil
#import zipfile

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

    logger = logging.getLogger('onnx2fluid')
    debug = kwargs.get('debug', False)

    # prepare arguments
    filename = kwargs['model'][0]
    basepath, _ = shutil.os.path.splitext(filename)
    save_dir = kwargs.get('output_dir', '')
    # model.onnx -> model/
    save_dir = (save_dir.rstrip(shutil.os.sep)
                if save_dir else basepath) + shutil.os.sep
    model_basename = DEFAULT_MODEL_MODULE + '.py'
    model_func_name = DEFAULT_MODEL_FUNC
    embed_params = kwargs.get('embed_params', False)
    onnx_opset_version = DEFAULT_ONNX_OPSET_VERSION
    onnx_opset_pedantic = kwargs.get('pedantic', True)
    onnx_skip_version_conversion = kwargs.get('skip_version_conversion', False)
    archive = kwargs.get('archive', None)

    # convert
    convert(
        filename,
        save_dir,
        model_basename=model_basename,
        model_func_name=model_func_name,
        embed_params=embed_params,
        onnx_opset_version=onnx_opset_version,
        onnx_opset_pedantic=onnx_opset_pedantic,
        onnx_skip_version_conversion=onnx_skip_version_conversion,
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
        decimal = kwargs.get('precision', 3.)

        logger.info('starting validation on desc ...')
        passed &= validate(
            shutil.os.path.join(save_dir, '__model__'),
            golden_data_filename,
            decimal=decimal,
        )

        logger.info('starting validation on code ...')
        passed &= validate(
            shutil.os.path.join(save_dir, model_basename),
            golden_data_filename,
            model_func_name=model_func_name,
            decimal=decimal,
            save_inference_model=debug,  # this overwrite desc file for test
        )

    if not passed:
        logger.error('validation failed, exit')
        return

    # create zip file
    if archive is not None:
        if archive == '':
            archive = save_dir.rstrip(shutil.os.sep) + '.zip'
        logger.info('compressing file to %s ...', archive)
        shutil.sys.stderr.write('\n')
        shutil.sys.stderr.flush()
        file_list = shutil.os.listdir(save_dir)
        fz = zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_LZMA)
        for idx, fn in enumerate(file_list):
            shutil.sys.stderr.write('\033[F\033[2K')
            logger.info('file {}/{}: {}'.format(idx + 1, len(file_list), fn))
            shutil.sys.stderr.flush()
            fz.write(shutil.os.path.join(save_dir, fn), arcname=fn)
        fz.close()
        logger.info('compressing done')


if __name__ == '__main__':
    logging.basicConfig(
        format=
        '[%(levelname)8s]%(name)s::%(funcName)s:%(lineno)04d: %(message)s',
        level=logging.DEBUG,
    )

    main(
        model=['../examples/t1.onnx'],
        output_dir='/tmp/export/',
        embed_params=False,
        pedantic=False,
        test_data='../examples/t1.npz',
        debug=True)

    main(
        model=['../examples/inception_v2/model.onnx'],
        output_dir='/tmp/export/',
        embed_params=True,
        pedantic=False,
        skip_version_conversion=False,
        test_data='../examples/inception_v2/test_data_set_2.npz',
        debug=True)
