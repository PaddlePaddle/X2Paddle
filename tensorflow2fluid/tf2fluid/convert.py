#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle_emitter import PaddleEmitter
from tensorflow_parser import TensorflowCkptParser
from tensorflow_parser import TensorflowPbParser
from six import text_type as _text_type
from utils import *
import argparse
import logging
import os
logging.basicConfig(level=logging.DEBUG)


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta_file",
        "-m",
        type=_text_type,
        default=None,
        help="meta file path for checkpoint format")
    parser.add_argument(
        "--ckpt_dir",
        "-c",
        type=_text_type,
        default=None,
        help="checkpoint directory")
    parser.add_argument(
        "--pb_file",
        "-p",
        type=_text_type,
        default=None,
        help="pb model file path")
    parser.add_argument(
        "--in_nodes",
        "-i",
        type=_text_type,
        nargs="+",
        default=None,
        help="input nodes name")
    parser.add_argument(
        "--input_shape",
        "-is",
        type=_text_type,
        nargs="+",
        default=None,
        help="input tensor shape")
    parser.add_argument(
        "--output_nodes",
        "-o",
        type=_text_type,
        nargs="+",
        default=None,
        help="output nodes name")
    parser.add_argument(
        "--save_dir",
        "-s",
        type=_text_type,
        default=None,
        help="path to save transformed paddle model")
    parser.add_argument(
        "--input_format",
        "-sf",
        type=_text_type,
        default=None,
        help="input data format(NHWC/NCHW or OTHER)")
    parser.add_argument(
        "--use_cuda",
        "-u",
        type=_text_type,
        default="True",
        help="True for use gpu")
    return parser


def run(args):
    if args.meta_file is None and args.pb_file is None:
        raise Exception("Need to define --meta_file or --pb_file")
    if args.input_format is None:
        raise Exception("Input format need to be defined(NHWC, NCHW or OTHER)")
    assert args.use_cuda == "True" or args.use_cuda == "False"
    if args.use_cuda == "False":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.input_format == "NHWC":
        input_format = NHWC
    elif args.input_format == "NCHW":
        input_format = NCHW
    elif args.input_format == "OTHER":
        input_format = OTHER
    else:
        raise Exception("Can not identify input format(NHWC/NCHW/OTHER)")

    assert args.in_nodes is not None
    assert args.output_nodes is not None
    assert args.input_shape is not None
    assert args.save_dir is not None

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    input_shape = list()
    for shape_str in args.input_shape:
        items = shape_str.split(',')
        for i in range(len(items)):
            if items[i] != "None":
                items[i] = int(items[i])
            else:
                items[i] = None

        input_shape.append(items)

    logging.info("Loading tensorflow model...")
    if args.meta_file is not None:
        parser = TensorflowCkptParser(args.meta_file, args.ckpt_dir,
                                      args.output_nodes, input_shape,
                                      args.in_nodes, input_format)
    else:
        parser = TensorflowPbParser(args.pb_file, args.output_nodes,
                                    input_shape, args.in_nodes, input_format)
    logging.info("Tensorflow model loaded!")

    emitter = PaddleEmitter(parser, args.save_dir)
    emitter.run()

    open(args.save_dir + "/__init__.py", "w").close()


def _main():
    parser = _get_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    _main()
