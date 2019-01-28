from paddle_emitter import PaddleEmitter
from tensorflow_parser import TensorflowCkptParser
from tensorflow_parser import TensorflowPbParser
from six import text_type as _text_type
import argparse
import sys
import os

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_file", "-m", type=_text_type, default=None, help="meta file path for checkpoint format")
    parser.add_argument("--ckpt_dir", "-c", type=_text_type, default=None, help="checkpoint directory")
    parser.add_argument("--pb_file", "-p", type=_text_type, default=None, help="pb model file path")
    parser.add_argument("--in_nodes", "-i", type=_text_type, nargs="+", default=None, help="input nodes name")
    parser.add_argument("--input_shape", "-is", type=_text_type, nargs="+", default=None, help="input tensor shape")
    parser.add_argument("--output_nodes", "-o", type=_text_type, nargs="+", default=None, help="output nodes name")
    parser.add_argument("--save_dir", "-s", type=_text_type, default=None, help="path to save transformed paddle model")
    parser.add_argument("--version", "-v", action="version", version="tensorflow2fluid version=0.0.1 Release @2019.01.28")
    return parser

def _convert(args):
    if args.meta_file is None and args.pb_file is None:
        raise Exception("Need to define --meta_file or --pb_file")
    assert args.in_nodes is not None
    assert args.output_nodes is not None
    assert args.input_shape is not None
    assert args.save_dir is not None

    if os.path.exists(args.save_dir):
        sys.stderr.write("save_dir already exists, change to a new path\n")
        return
    
    os.makedirs(args.save_dir)

    input_shape = list()
    for shape_str in args.input_shape:
        items = shape_str.split(',')
        for i in range(len(items)):
            if items[i] != "None":
                items[i] = int(items[i])
        input_shape.append(items)

    sys.stderr.write("\nLoading tensorflow model......\n")
    if args.meta_file is not None:
        parser = TensorflowCkptParser(args.meta_file, args.ckpt_dir, args.output_nodes, input_shape, args.in_nodes)
    else:
        parser = TensorflowPbParser(args.pb_file, args.output_nodes, input_shape, args.in_nodes)
    sys.stderr.write("Tensorflow model loaded!\n")
    emitter = PaddleEmitter(parser, args.save_dir)
    emitter.run()

    open(args.save_dir+"/__init__.py", "w").close()

def _main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = _get_parser()
    args = parser.parse_args()
    _convert(args)

if __name__ == "__main__":
    _main()
