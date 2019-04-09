# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件允许模块包以python -m onnx2fluid方式直接执行。

Authors: Macrobull
Date:    2019/02/22 10:25:46
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import argparse, logging, sys
import argparse
import logging
import sys

parser = argparse.ArgumentParser(
    description='onnx2fluid',
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
    '--test_data',
    '-t',
    type=str,
    default='',
    help='I/O golden data for validation, e.g. test.npy, test.npz',
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
parser.add_argument(
    '--archive',
    '-z',
    nargs='?',
    type=str,
    default=None,
    const='',
    help='compress outputs to ZIP file if conversion successed',
)
parser.add_argument(
    '--precision',
    '-p',
    type=float,
    default=3.,
    help='assertion decimal for validation',
)
args = parser.parse_args()

logging_format = '[%(levelname)8s]%(name)s::%(funcName)s:%(lineno)04d: %(message)s'
logging_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(format=logging_format, level=logging_level)

try:
    from . import cmdline
except ImportError:
    import cmdline

# imports
main = cmdline.main

sys.exit(main(**args.__dict__))
