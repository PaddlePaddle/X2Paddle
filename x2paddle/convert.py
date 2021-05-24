# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
from x2paddle import program
import argparse
import sys


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=_text_type,
        default=None,
        help="define model file path for tensorflow or onnx")
    parser.add_argument(
        "--prototxt",
        "-p",
        type=_text_type,
        default=None,
        help="prototxt file of caffe model")
    parser.add_argument(
        "--weight",
        "-w",
        type=_text_type,
        default=None,
        help="weight file of caffe model")
    parser.add_argument(
        "--save_dir",
        "-s",
        type=_text_type,
        default=None,
        help="path to save translated model")
    parser.add_argument(
        "--framework",
        "-f",
        type=_text_type,
        default=None,
        help="define which deeplearning framework(tensorflow/caffe/onnx/paddle2onnx)"
    )
    parser.add_argument(
        "--caffe_proto",
        "-c",
        type=_text_type,
        default=None,
        help="optional: the .py file compiled by caffe proto file of caffe model"
    )
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        default=False,
        help="get version of x2paddle")
    parser.add_argument(
        "--define_input_shape",
        "-d",
        action="store_true",
        default=False,
        help="define input shape for tf model")
    parser.add_argument(
        "--convert_torch_project",
        "-tp",
        action='store_true',
        help="Convert the PyTorch Project.")
    parser.add_argument(
        "--project_dir",
        "-pd",
        type=_text_type,
        default=None,
        help="define project folder path for pytorch")
    parser.add_argument(
        "--pretrain_model",
        "-pm",
        type=_text_type,
        default=None,
        help="pretrain model file of pytorch model")

    return parser


def tf2paddle(model_path, save_dir, define_input_shape=False):
    # check tensorflow installation and version
    try:
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
        import tensorflow as tf
        version = tf.__version__
        if version >= '2.0.0' or version < '1.0.0':
            print(
                "[ERROR] 1.0.0<=tensorflow<2.0.0 is required, and v1.14.0 is recommended"
            )
            return
    except:
        print(
            "[ERROR] Tensorflow is not installed, use \"pip install tensorflow\"."
        )
        return

    from x2paddle.decoder.tf_decoder import TFDecoder
    from x2paddle.op_mapper.tf2paddle.tf_op_mapper import TFOpMapper

    print("Now translating model from tensorflow to paddle.")
    model = TFDecoder(model_path, define_input_shape=define_input_shape)
    mapper = TFOpMapper(model)
    mapper.paddle_graph.build()
    from x2paddle.optimizer.optimizer import GraphOptimizer
    graph_opt = GraphOptimizer(source_frame="tf")
    graph_opt.optimize(mapper.paddle_graph)
    mapper.paddle_graph.gen_model(save_dir)


def caffe2paddle(proto, weight, save_dir, caffe_proto):
    from x2paddle.decoder.caffe_decoder import CaffeDecoder
    from x2paddle.op_mapper.caffe2paddle.caffe_op_mapper import CaffeOpMapper
    import google.protobuf as gpb
    ver_part = gpb.__version__.split('.')
    version_satisfy = False
    if (int(ver_part[0]) == 3 and int(ver_part[1]) >= 6) \
        or (int(ver_part[0]) > 3):
        version_satisfy = True
    assert version_satisfy, '[ERROR] google.protobuf >= 3.6.0 is required'
    print("Now translating model from caffe to paddle.")
    model = CaffeDecoder(proto, weight, caffe_proto)
    mapper = CaffeOpMapper(model)
    mapper.paddle_graph.build()
    print("Model optimizing ...")
    from x2paddle.optimizer.optimizer import GraphOptimizer
    graph_opt = GraphOptimizer(source_frame="caffe")
    graph_opt.optimize(mapper.paddle_graph)
    print("Model optimized.")
    mapper.paddle_graph.gen_model(save_dir)


def onnx2paddle(model_path, save_dir):
    # check onnx installation and version
    try:
        import onnx
        version = onnx.version.version
        if version < '1.6.0':
            print("[ERROR] onnx>=1.6.0 is required")
            return
    except:
        print("[ERROR] onnx is not installed, use \"pip install onnx==1.6.0\".")
        return
    print("Now translating model from onnx to paddle.")

    from x2paddle.decoder.onnx_decoder import ONNXDecoder
    from x2paddle.op_mapper.onnx2paddle.onnx_op_mapper import ONNXOpMapper
    model = ONNXDecoder(model_path)
    mapper = ONNXOpMapper(model)
    mapper.paddle_graph.build()
    mapper.paddle_graph.gen_model(save_dir)


def pytorch2paddle(module, save_dir, jit_type="trace", input_examples=None):
    # check pytorch installation and version
    try:
        import torch
        version = torch.__version__
        ver_part = version.split('.')
        print(ver_part)
        if int(ver_part[1]) < 5:
            print("[ERROR] pytorch>=1.5.0 is required")
            return
    except:
        print(
            "[ERROR] Pytorch is not installed, use \"pip install torch==1.5.0 torchvision\"."
        )
        return
    print("Now translating model from pytorch to paddle.")

    from x2paddle.decoder.pytorch_decoder import ScriptDecoder, TraceDecoder
    from x2paddle.op_mapper.pytorch2paddle.pytorch_op_mapper import PyTorchOpMapper

    if jit_type == "trace":
        model = TraceDecoder(module, input_examples)
    else:
        model = ScriptDecoder(module, input_examples)
    mapper = PyTorchOpMapper(model)
    mapper.paddle_graph.build()
    print("Model optimizing ...")
    from x2paddle.optimizer.optimizer import GraphOptimizer
    graph_opt = GraphOptimizer(source_frame="pytorch", jit_type=jit_type)
    graph_opt.optimize(mapper.paddle_graph)
    print("Model optimized.")
    mapper.paddle_graph.gen_model(save_dir, jit_type=jit_type)


def main():
    if len(sys.argv) < 2:
        print("Use \"x2paddle -h\" to print the help information")
        print("For more information, please follow our github repo below:)")
        print("\nGithub: https://github.com/PaddlePaddle/X2Paddle.git\n")
        return

    parser = arg_parser()
    args = parser.parse_args()

    if args.version:
        import x2paddle
        print("x2paddle-{} with python>=3.5, paddlepaddle>=1.6.0\n".format(
            x2paddle.__version__))
        return

    if not args.convert_torch_project:
        assert args.framework is not None, "--framework is not defined(support tensorflow/caffe/onnx)"
    assert args.save_dir is not None, "--save_dir is not defined"

    try:
        import platform
        v0, v1, v2 = platform.python_version().split('.')
        if not (int(v0) >= 3 and int(v1) >= 5):
            print("[ERROR] python>=3.5 is required")
            return
        import paddle
        v0, v1, v2 = paddle.__version__.split('.')
        print("paddle.__version__ = {}".format(paddle.__version__))
        if v0 == '0' and v1 == '0' and v2 == '0':
            print("[WARNING] You are use develop version of paddlepaddle")
        elif int(v0) != 2 or int(v1) < 0:
            print("[ERROR] paddlepaddle>=2.0.0 is required")
            return
    except:
        print(
            "[ERROR] paddlepaddle not installed, use \"pip install paddlepaddle\""
        )

    if args.convert_torch_project:
        assert args.project_dir is not None, "--project_dir should be defined while translating pytorch project"
        from x2paddle.project_convertor.pytorch.convert import main as convert_torch
        convert_torch(args)
    else:
        if args.framework == "tensorflow":
            assert args.model is not None, "--model should be defined while translating tensorflow model"
            define_input_shape = False
            if args.define_input_shape:
                define_input_shape = True
            tf2paddle(args.model, args.save_dir, define_input_shape)

        elif args.framework == "caffe":
            assert args.prototxt is not None and args.weight is not None, "--prototxt and --weight should be defined while translating caffe model"
            caffe2paddle(args.prototxt, args.weight, args.save_dir,
                         args.caffe_proto)
        elif args.framework == "onnx":
            assert args.model is not None, "--model should be defined while translating onnx model"
            onnx2paddle(args.model, args.save_dir)
        elif args.framework == "paddle2onnx":
            print(
                "Paddle to ONNX tool has been migrated to the new github: https://github.com/PaddlePaddle/paddle2onnx"
            )

        else:
            raise Exception(
                "--framework only support tensorflow/caffe/onnx now")


if __name__ == "__main__":
    main()
