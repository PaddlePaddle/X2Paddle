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


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        "-m",
                        type=_text_type,
                        default=None,
                        help="model file path")
    parser.add_argument("--proto",
                        "-p",
                        type=_text_type,
                        default=None,
                        help="proto file of caffe model")
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
    return parser


def tf2paddle(model, save_dir):
    print("Now translating model from tensorflow to paddle.")
    from x2paddle.parser.tf_parser import TFParser
    from x2paddle.optimizer.tf_optimizer import TFGraphOptimizer
    from x2paddle.emitter.tf_emitter import TFEmitter
    parser = TFParser(model)
    emitter = TFEmitter(parser)
    emitter.run()
    emitter.save_python_model(save_dir)


def caffe2paddle(proto, weight, save_dir):
    print("Now translating model from caffe to paddle.")
    from x2paddle.parser.caffe_parser import CaffeParser
    from x2paddle.emitter.caffe_emitter import CaffeEmitter
    parser = CaffeParser(proto, weight)
    emitter = CaffeEmitter(parser)
    emitter.run()
    emitter.save_python_model(save_dir)


def main():
    parser = arg_parser()
    args = parser.parse_args()

    assert args.framework is not None, "--from is not defined(tensorflow/caffe)"
    assert args.save_dir is not None, "--save_dir is not defined"

    if args.framework == "tensorflow":
        assert args.model is not None, "--model should be defined while translate tensorflow model"
        tf2paddle(args.model, args.save_dir)

    elif args.framework == "caffe":
        assert args.proto is not None, "--proto and --weight should be defined while translate caffe model"
        caffe2paddle(args.proto, args.weight, args.save_dir)

    else:
        raise Exception("--framework only support tensorflow/caffe now")


if __name__ == "__main__":
    main()
