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
                        help="define model file path for tensorflow or onnx")
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
    parser.add_argument(
        "--framework",
        "-f",
        type=_text_type,
        default=None,
        help="define which deeplearning framework(tensorflow/caffe/onnx)")
    parser.add_argument(
        "--caffe_proto",
        "-c",
        type=_text_type,
        default=None,
        help="optional: the .py file compiled by caffe proto file of caffe model"
    )
    parser.add_argument("--version",
                        "-v",
                        action="store_true",
                        default=False,
                        help="get version of x2paddle")
    parser.add_argument(
        "--without_data_format_optimization",
        "-wo",
        action="store_true",
        default=False,
        help="tf model conversion without data format optimization")
    parser.add_argument("--define_input_shape",
                        "-d",
                        action="store_true",
                        default=False,
                        help="define input shape for tf model")

    return parser


def tf2paddle(model_path,
              save_dir,
              without_data_format_optimization=False,
              define_input_shape=False):
    # check tensorflow installation and version
    try:
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
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
    from x2paddle.op_mapper.tf_op_mapper_nhwc import TFOpMapperNHWC
    from x2paddle.optimizer.tf_optimizer import TFOptimizer

    print("Now translating model from tensorflow to paddle.")
    model = TFDecoder(model_path, define_input_shape=define_input_shape)
    if not without_data_format_optimization:
        mapper = TFOpMapper(model)
        optimizer = TFOptimizer(mapper)
        # neccesary optimization
        optimizer.delete_redundance_code()
        # optimizer below is experimental
        optimizer.optimize_elementwise_op()
        optimizer.merge_activation()
        optimizer.merge_bias()
        optimizer.optimize_sub_graph()


#        optimizer.merge_batch_norm()
#        optimizer.merge_prelu()
    else:
        mapper = TFOpMapperNHWC(model)
        optimizer = TFOptimizer(mapper)
        optimizer.delete_redundance_code()
        optimizer.strip_graph()
        optimizer.merge_activation()
        optimizer.merge_bias()
        optimizer.make_nchw_input_output()
        optimizer.remove_transpose()
    mapper.save_inference_model(save_dir)


def caffe2paddle(proto, weight, save_dir, caffe_proto):
    from x2paddle.decoder.caffe_decoder import CaffeDecoder
    from x2paddle.op_mapper.caffe_op_mapper import CaffeOpMapper
    from x2paddle.optimizer.caffe_optimizer import CaffeOptimizer
    import google.protobuf as gpb
    ver_str = gpb.__version__.replace('.', '')
    ver_int = int(ver_str[0:2])
    assert ver_int >= 36, 'The version of protobuf must be larger than 3.6.0!'
    print("Now translating model from caffe to paddle.")
    model = CaffeDecoder(proto, weight, caffe_proto)
    mapper = CaffeOpMapper(model)
    optimizer = CaffeOptimizer(mapper)
    optimizer.merge_bn_scale()
    optimizer.merge_op_activation()
    mapper.save_inference_model(save_dir)


def onnx2paddle(model_path, save_dir):
    # check onnx installation and version
    try:
        import onnx
        version = onnx.version.version
        if version != '1.5.0':
            print("onnx==1.5.0 is required")
            return
    except:
        print("onnx is not installed, use \"pip install onnx==1.5.0\".")
        return
    print("Now translating model from onnx to paddle.")

    from x2paddle.decoder.onnx_decoder import ONNXDecoder
    model = ONNXDecoder(model_path)

    from x2paddle.op_mapper.onnx_op_mapper import ONNXOpMapper
    mapper = ONNXOpMapper(model, save_dir)

    from x2paddle.optimizer.onnx_optimizer import ONNXOptimizer
    optimizer = ONNXOptimizer(mapper)

    optimizer.delete_redundance_code()
    mapper.save_inference_model(save_dir)


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
        print("x2paddle-{} with python>=3.5, paddlepaddle>=1.5.0\n".format(
            x2paddle.__version__))
        return

    assert args.framework is not None, "--framework is not defined(support tensorflow/caffe/onnx)"
    assert args.save_dir is not None, "--save_dir is not defined"

    try:
        import paddle
        v0, v1, v2 = paddle.__version__.split('.')
        if int(v0) != 1 or int(v1) < 5:
            print("paddlepaddle>=1.5.0 is required")
            return
    except:
        print("paddlepaddle not installed, use \"pip install paddlepaddle\"")

    if args.framework == "tensorflow":
        assert args.model is not None, "--model should be defined while translating tensorflow model"
        without_data_format_optimization = False
        define_input_shape = False
        if args.without_data_format_optimization:
            without_data_format_optimization = True
        if args.define_input_shape:
            define_input_shape = True
        tf2paddle(args.model, args.save_dir, without_data_format_optimization,
                  define_input_shape)

    elif args.framework == "caffe":
        assert args.prototxt is not None and args.weight is not None, "--prototxt and --weight should be defined while translating caffe model"
        caffe2paddle(args.prototxt, args.weight, args.save_dir,
                     args.caffe_proto)
    elif args.framework == "onnx":
        assert args.model is not None, "--model should be defined while translating onnx model"
        onnx2paddle(args.model, args.save_dir)
    else:
        raise Exception("--framework only support tensorflow/caffe/onnx now")


if __name__ == "__main__":
    main()
