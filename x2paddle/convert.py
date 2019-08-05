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

from six import text_type as _text_type
import argparse
import sys


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        "-m",
                        type=_text_type,
                        default=None,
                        help="model file path")
    parser.add_argument("--prototxt",
                        "-p",
                        type=_text_type,
                        default=None,
                        help="prototxt file of caffe model")
    parser.add_argument("--weight",
                        "-w",
                        type=_text_type,
                        default=None,
                        help="weight file of caffe model")
    parser.add_argument("--save_dir",
                        "-s",
                        type=_text_type,
                        default=None,
                        help="path to save translated model")
    parser.add_argument("--framework",
                        "-f",
                        type=_text_type,
                        default=None,
                        help="define which deeplearning framework")
    parser.add_argument("--caffe_proto",
                        "-c",
                        type=_text_type,
                        default=None,
                        help="caffe proto file of caffe model")
    return parser


def tf2paddle(model_path, save_dir):
    # check tensorflow installation and version
    try:
        import tensorflow as tf
        version = tf.__version__
        if version >= '2.0.0' or version < '1.0.0':
            print(
                "1.0.0<=tensorflow<2.0.0 is required, and v1.14.0 is recommended"
            )
            return
    except:
        print("Tensorflow is not installed, use \"pip install tensorflow\".")
        return

    from x2paddle.decoder.tf_decoder import TFDecoder
    from x2paddle.op_mapper.tf_op_mapper import TFOpMapper
    from x2paddle.optimizer.tf_optimizer import TFOptimizer

    print("Now translating model from tensorflow to paddle.")
    model = TFDecoder(model_path)
    mapper = TFOpMapper(model)
    optimizer = TFOptimizer(mapper)
    # neccesary optimization
    optimizer.delete_redundance_code()
    # optimizer below is experimental
    optimizer.merge_activation()
    optimizer.merge_bias()
    mapper.save_inference_model(save_dir)


def caffe2paddle(proto, weight, save_dir, caffe_proto):
    if caffe_proto is not None:
        import os
        if not os.path.isfile(caffe_proto + 'caffe_pb2.py'):
            print("The file that resolve caffe is not exist.")
            return
    else:
        try:
            import caffe
            version = caffe.__version__
            if version != '1.0.0':
                print("caffe == 1.0.0 is required")
                return
        except:
            print("Caffe is not installed.")
            print(
                "You have 2 options: 1. install caffe 2. compile the caffe.proto"
            )
            return
    from x2paddle.decoder.caffe_decoder import CaffeDecoder
    from x2paddle.op_mapper.caffe_op_mapper import CaffeOpMapper

    print("Now translating model from caffe to paddle.")
    model = CaffeDecoder(proto, weight, caffe_proto)
    mapper = CaffeOpMapper(model)
    mapper.save_inference_model(save_dir)


def main():
    if len(sys.argv) < 2:
        print("Use \"x2paddle -h\" to print the help information")
        return

    parser = arg_parser()
    args = parser.parse_args()

    try:
        import paddle.fluid as fluid
        import paddle
        v0, v1, v2 = paddle.__version__.split('.')
        if int(v0) != 1 or int(v1) < 5:
            print("paddlepaddle>=1.5.0 is required")
            return
    except:
        print("paddlepaddle not installed, use \"pip install paddlepaddle\"")

    assert args.framework is not None, "--from is not defined(tensorflow/caffe)"
    assert args.save_dir is not None, "--save_dir is not defined"

    if args.framework == "tensorflow":
        assert args.model is not None, "--model should be defined while translating tensorflow model"
        tf2paddle(args.model, args.save_dir)

    elif args.framework == "caffe":
        assert args.prototxt is not None and args.weight is not None, "--prototxt and --weight should be defined while translating caffe model"
        caffe2paddle(args.prototxt, args.weight, args.save_dir,
                     args.caffe_proto)

    else:
        raise Exception("--framework only support tensorflow/caffe now")


if __name__ == "__main__":
    main()
