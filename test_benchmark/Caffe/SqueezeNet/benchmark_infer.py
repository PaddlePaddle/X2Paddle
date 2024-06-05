import os
import argparse
import pickle
import numpy as np
import sys
sys.path.append('../tools/')

from predict import BenchmarkPipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Model inference')
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=1)
    def str2bool(v):
        return v.lower() in ("true", "t", "1")
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--enable_trt", type=str2bool, default=True, help="enable trt")
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=True)
    return parser.parse_args()

def main(args):
    with open('caffe_input.pkl', 'rb') as inp:
        input_data = pickle.load(inp)["data0"]
    with open('caffe_output.pkl', 'rb') as oup:
        caffe_result = pickle.load(oup)["prob"]
    caffe_result = np.array(caffe_result)
    benchmark_pipeline = BenchmarkPipeline(model_dir="pd_model_dygraph/inference_model/",
                                           model_name='SqueezeNet',
                                           use_gpu=args.use_gpu,
                                           enable_trt=args.enable_trt,
                                           cpu_threads=args.cpu_threads,
                                           enable_mkldnn=args.enable_mkldnn)
    benchmark_pipeline.run_benchmark(data=input_data, caffe_result=caffe_result, warmup=1, repeats=1)
    benchmark_pipeline.analysis_operators(model_dir="pd_model_dygraph/inference_model/")
    benchmark_pipeline.report()
    
if __name__ == '__main__':
    args = parse_args()
    main(args)