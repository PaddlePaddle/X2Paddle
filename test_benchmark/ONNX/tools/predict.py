import argparse
import codecs
import os
import time
import sys

import pynvml
import psutil
import GPUtil
import yaml
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config
from benchmark_utils import PaddleInferBenchmark


def load_predictor(model_dir,
                   batch_size=1,
                   use_gpu=False,
                   enable_trt=False,
                   cpu_threads=1,
                   enable_mkldnn=False,
                   enable_dynamic_shape=False):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        use_gpu (bool): whether use gpu
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    """
    config = Config(os.path.join(model_dir, 'model.pdmodel'),
                    os.path.join(model_dir, 'model.pdiparams'))
    if use_gpu:
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
        if enable_trt:
            try:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=PrecisionType.Float32,
                    max_batch_size=1,
                    min_subgraph_size=3)
                if enable_dynamic_shape:
                    config.collect_shape_range_info("shape.pbtxt")
                    config.enable_tuned_tensorrt_dynamic_shape(
                        "shape.pbtxt", True)
            except Exception as e:
                print(
                    "The current environment does not support `tensorrt`, so disable tensorrt."
                )
                pass
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    pass_builder = config.pass_builder()
    predictor = create_predictor(config)
    return predictor, config


class BenchmarkPipeline:

    def __init__(self,
                 model_dir,
                 model_name,
                 batch_size=1,
                 use_gpu=True,
                 enable_trt=True,
                 cpu_threads=1,
                 enable_mkldnn=True,
                 enable_dynamic_shape=False):
        self.predictor, self.config = load_predictor(
            model_dir,
            batch_size=batch_size,
            use_gpu=use_gpu,
            enable_trt=enable_trt,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            enable_dynamic_shape=enable_dynamic_shape)
        self.model_name = model_name
        self.batch_size = batch_size

    def run_benchmark(self,
                      data=None,
                      onnx_result=None,
                      warmup=0,
                      repeats=1,
                      abs_diff_threshold=1e-05,
                      relative_diff_threshold=1e-05):
        if repeats == 0:
            assert 'repeat nums should greater than 0'
        results = None
        self.inference_time = Times()
        cpu_mem, gpu_mem = 0, 0
        gpu_id = 0
        gpu_util = 0

        input_names = self.predictor.get_input_names()
        if type(data) is np.ndarray:
            for i in range(len(input_names)):
                input_tensor = self.predictor.get_input_handle(input_names[i])
                input_tensor.copy_from_cpu(data)
        elif isinstance(data, dict):
            input_data = data
            for i in range(len(input_names)):
                input_tensor = self.predictor.get_input_handle(input_names[i])
                data = input_data[list(input_data.keys())[i]]
                input_tensor.copy_from_cpu(data)
        elif isinstance(data, list):
            input_data = data
            if type(input_data[0]) is np.ndarray:
                for i in range(len(input_names)):
                    input_tensor = self.predictor.get_input_handle(
                        input_names[i])
                    data = input_data[i]
                    input_tensor.copy_from_cpu(data)
            else:
                for i in range(len(input_names)):
                    input_tensor = self.predictor.get_input_handle(
                        input_names[i])
                    data = input_data[0][list(input_data[0].keys())[i]]
                    input_tensor.copy_from_cpu(data)
        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            results = []
            for output_name in output_names:
                output_tensor = self.predictor.get_output_handle(output_name)
                results.append(output_tensor.copy_to_cpu())
        for i in range(repeats):
            self.inference_time.start()
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            results = []
            for output_name in output_names:
                output_tensor = self.predictor.get_output_handle(output_name)
                results.append(output_tensor.copy_to_cpu())
            self.inference_time.end()
            gpu_util += get_current_gputil(gpu_id)
            cm, gm = get_current_memory_mb(gpu_id)
            cpu_mem += cm
            gpu_mem += gm

        # compare onnx_result and paddle_inference_result
        self.compare_result = "Compare Successed"
        if isinstance(onnx_result, list):
            for i in range(len(onnx_result)):
                # avoid topk diff
                if self.model_name == "mmdetection_SSD" or self.model_name == "SwinTransformer":
                    diff = results[i][:, :20] - onnx_result[i][:, :20]
                elif self.model_name == "mmdetection_Faster-RCNN":
                    diff = results[i][:, :15] - onnx_result[i][:, :15]
                else:
                    diff = results[i] - onnx_result[i]
                max_abs_diff = np.fabs(diff).max()
                if max_abs_diff >= abs_diff_threshold:
                    #                     relative_diff_all = np.fabs(diff) / np.fabs(results[i])
                    relative_diff_all = max_abs_diff / np.fabs(
                        onnx_result[i]).max()
                    relative_diff = relative_diff_all.max()
                    if relative_diff >= relative_diff_threshold:
                        self.compare_result = "Compare Failed"
        else:
            diff = results - onnx_result
            max_abs_diff = np.fabs(diff).max()

            if max_abs_diff < abs_diff_threshold:
                self.compare_result = "Compare Successed"
            else:
                relative_diff = max_abs_diff / np.fabs(onnx_result).max()
                if relative_diff < relative_diff_threshold:
                    self.compare_result = "Compare Successed"
                else:
                    self.compare_result = "Compare Failed"
        self.avg_inference = self.inference_time.value() / repeats
        self.avg_cpu_mem = cpu_mem / repeats
        self.avg_gpu_mem = gpu_mem / repeats
        self.avg_gpu_util = gpu_util / repeats

    def analysis_operators(self, model_dir):
        paddle.enable_static()
        exe = paddle.static.Executor(paddle.CPUPlace())
        # test Dygraph
        [prog, inputs, outputs
         ] = fluid.io.load_inference_model(dirname=model_dir,
                                           executor=exe,
                                           model_filename="model.pdmodel",
                                           params_filename="model.pdiparams")
        #test op nums
        op_dict = dict()
        op_nums = 0
        for i, op in enumerate(prog.block(0).ops):
            if op.type in ['feed', 'fetch']:
                continue
            paddle_op_type = str(op.type)
            if paddle_op_type in op_dict:
                op_dict[paddle_op_type] += 1
            else:
                op_dict[paddle_op_type] = 1
            op_nums += 1
        self.op_nums = op_nums
        self.op_classes = len(op_dict)

    def report(self):
        perf_info = {
            'inference_time_s': self.avg_inference,
        }
        model_info = {
            'model_name': self.model_name,
        }
        data_info = {
            'batch_size': self.batch_size,
        }
        resource_info = {
            'cpu_rss_mb': self.avg_cpu_mem,
            'gpu_rss_mb': self.avg_gpu_mem,
            'gpu_util': self.avg_gpu_util
        }
        op_info = {'op_nums': self.op_nums, 'op_classes': self.op_classes}
        compare_info = {
            'compare_result': self.compare_result,
        }
        x2paddle_log = PaddleInferBenchmark(self.config, model_info, data_info,
                                            perf_info, resource_info, op_info,
                                            compare_info)
        x2paddle_log('X2paddle')


# create time count class
class Times(object):

    def __init__(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def start(self):
        self.st = time.time()

    def end(self, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += self.et - self.st
        else:
            self.time = self.et - self.st

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 4)


def get_current_memory_mb(gpu_id=None):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024. / 1024.
    gpu_mem = 0
    if gpu_id is not None:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024. / 1024.
    return cpu_mem, gpu_mem


def get_current_gputil(gpu_id):
    GPUs = GPUtil.getGPUs()
    gpu_load = GPUs[gpu_id].load
    return gpu_load


def main():
    benchmarkpipeline = BenchmarkPipeline(
        model_dir="mmdetection_yolov3_paddle_test_0519/inference_model/",
        model_name="yolov3")
    data = np.load("real_img_data_yolo_50.npy")
    benchmarkpipeline.run_benchmark(data=data, warmup=10, repeats=10)
    benchmarkpipeline.report()


if __name__ == '__main__':
    main()
