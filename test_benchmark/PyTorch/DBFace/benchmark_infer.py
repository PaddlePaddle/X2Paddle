import os
import argparse
import pickle
import common
import numpy as np
import sys

sys.path.append('../tools/')

from predict import BenchmarkPipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Model inference')
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        help='Mini batch size of one gpu or cpu.',
                        type=int,
                        default=1)

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--enable_trt",
                        type=str2bool,
                        default=True,
                        help="enable trt")
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=True)
    return parser.parse_args()


def main(args):
    image = common.imread("../dataset/DBFace/datas/selfie.jpg")
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]
    image = common.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    data = np.expand_dims(image.transpose(2, 0, 1), axis=0)
    with open('../dataset/DBFace/outputs.pkl', 'rb') as oup:
        pytorch_output = pickle.load(oup)
    pytorch_result = list()
    pytorch_result.append(pytorch_output["hm"])
    pytorch_result.append(pytorch_output["box"])
    pytorch_result.append(pytorch_output["landmark"])
    benchmark_pipeline = BenchmarkPipeline(
        model_dir="pd_model_trace/inference_model/",
        model_name='DBFace_trace',
        use_gpu=args.use_gpu,
        enable_trt=args.enable_trt,
        cpu_threads=args.cpu_threads,
        enable_mkldnn=args.enable_mkldnn)
    benchmark_pipeline.run_benchmark(data=data,
                                     pytorch_result=pytorch_result,
                                     warmup=1,
                                     repeats=1)
    benchmark_pipeline.analysis_operators(
        model_dir="pd_model_trace/inference_model/")
    benchmark_pipeline.report()


if __name__ == '__main__':
    args = parse_args()
    main(args)
