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
        "--without_data_format_optimization",
        "-wo",
        type=_text_type,
        default="True",
        help="tf model conversion without data format optimization")
    parser.add_argument(
        "--define_input_shape",
        "-d",
        action="store_true",
        default=False,
        help="define input shape for tf model")
    parser.add_argument(
        "--params_merge",
        "-pm",
        action="store_true",
        default=False,
        help="define whether merge the params")
    parser.add_argument(
        "--input_shapes",
        "-is",
        action='append',
        default=None,
        help="define the inputs' shape")
    return parser


def tf2paddle(model_path,
              save_dir,
              without_data_format_optimization=False,
              define_input_shape=False,
              params_merge=False):
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
    from x2paddle import program
    from x2paddle.decoder.tf_decoder import TFDecoder
    from x2paddle.op_mapper.tf_op_mapper import TFOpMapper
    from x2paddle.optimizer.tensorflow.bias import BiasOpt
    from x2paddle.optimizer.tensorflow.transpose import TransposeOpt
    from x2paddle.optimizer.tensorflow.batch_norm import BatchNormOpt
    from x2paddle.optimizer.tensorflow.prelu import PReLUOpt

    print("Now translating model from tensorflow to paddle.")
    model = TFDecoder(model_path, define_input_shape=define_input_shape)
    mapper = TFOpMapper(model)
    program.build()
    bias_opt = BiasOpt()
    transpose_opt = TransposeOpt()
    batch_norm_opt = BatchNormOpt()
    prelu_opt = PReLUOpt()
    bias_opt.run(program)
    batch_norm_opt.run(program)
    prelu_opt.run(program)
    transpose_opt.run(program)
    program.gen_model(save_dir)


def caffe2paddle(proto, weight, save_dir, caffe_proto, params_merge=False):
    from x2paddle.decoder.caffe_decoder import CaffeDecoder
    from x2paddle.op_mapper.caffe_op_mapper import CaffeOpMapper
    from x2paddle.optimizer.caffe_optimizer import CaffeOptimizer
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
    optimizer = CaffeOptimizer(mapper)
    optimizer.merge_bn_scale()
    optimizer.merge_op_activation()
    mapper.save_inference_model(save_dir, params_merge)


def onnx2paddle(model_path, save_dir, params_merge=False):
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

    from x2paddle.op_mapper.onnx2paddle.onnx_op_mapper import ONNXOpMapper
    from x2paddle.decoder.onnx_decoder import ONNXDecoder
    from x2paddle.optimizer.onnx_optimizer import ONNXOptimizer
    model = ONNXDecoder(model_path)
    mapper = ONNXOpMapper(model)
    print("Model optimizing ...")
    optimizer = ONNXOptimizer(mapper)
    optimizer.delete_redundance_code()
    print("Model optimized.")

    print("Paddle model and code generating ...")
    mapper.save_inference_model(save_dir, params_merge)
    print("Paddle model and code generated.")


def pytorch2paddle(model_path, save_dir, input_shapes):
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

    from x2paddle.decoder.pytorch_decoder import PyTorchDecoder
    from x2paddle.op_mapper.pytorch2paddle import pytorch_op_mapper
    model = PyTorchDecoder(model_path)
    mapper = pytorch_op_mapper.PyTorchOpMapper(model)
    mapper.graph.build()
    print("Model optimizing ...")
    from x2paddle.optimizer.pytorch_optimizer.optimizer import GraphOptimizer
    graph_opt = GraphOptimizer()
    graph_opt.optimize(mapper.graph)
    print("Model optimized.")
    if input_shapes is not None:
        real_input_shapes = list()
        for shape in input_shapes:
            sp = shape[1:-1].split(",")
            for i, s in enumerate(sp):
                sp[i] = int(s)
            real_input_shapes.append(sp)
    else:
        real_input_shapes = None
    mapper.graph.gen_model(save_dir, real_input_shapes)


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

    assert args.framework is not None, "--framework is not defined(support tensorflow/caffe/onnx)"
    assert args.save_dir is not None, "--save_dir is not defined"

    try:
        import paddle
        v0, v1, v2 = paddle.__version__.split('.')
        print("paddle.__version__ = {}".format(paddle.__version__))
        if v0 == '0' and v1 == '0' and v2 == '0':
            print("[WARNING] You are use develop version of paddlepaddle")
        elif int(v0) != 1 or int(v1) < 6:
            print("[ERROR] paddlepaddle>=1.6.0 is required")
            return
    except:
        print(
            "[ERROR] paddlepaddle not installed, use \"pip install paddlepaddle\""
        )

    if args.framework == "tensorflow":
        assert args.model is not None, "--model should be defined while translating tensorflow model"
        assert args.without_data_format_optimization in [
            "True", "False"
        ], "--the param without_data_format_optimization should be defined True or False"
        define_input_shape = False
        params_merge = False
        without_data_format_optimization = True if args.without_data_format_optimization == "True" else False
        if args.define_input_shape:
            define_input_shape = True
        if args.params_merge:
            params_merge = True
        tf2paddle(args.model, args.save_dir, without_data_format_optimization,
                  define_input_shape, params_merge)

    elif args.framework == "caffe":
        assert args.prototxt is not None and args.weight is not None, "--prototxt and --weight should be defined while translating caffe model"
        params_merge = False
        if args.params_merge:
            params_merge = True
        caffe2paddle(args.prototxt, args.weight, args.save_dir,
                     args.caffe_proto, params_merge)
    elif args.framework == "onnx":
        assert args.model is not None, "--model should be defined while translating onnx model"
        params_merge = False

        if args.params_merge:
            params_merge = True
        onnx2paddle(args.model, args.save_dir, params_merge)
        
    elif args.framework == "paddle2onnx":
        print("Paddle to ONNX tool has been migrated to the new github: https://github.com/PaddlePaddle/paddle2onnx")

    else:
        raise Exception(
            "--framework only support tensorflow/caffe/onnx/ now")


if __name__ == "__main__":
    main()
