import os
import time
import pickle

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.inference import Config
from paddle.inference import create_predictor

def get_current_memory_mb():
    """
    It is used to Obtain the memory usage of the CPU and GPU during the running of the program.
    And this function Current program is time-consuming.
    """
    import pynvml
    import psutil
    import GPUtil
    gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', 0))

    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024. / 1024.
    gpu_mem = 0
    gpu_percent = 0
    gpus = GPUtil.getGPUs()
    if gpu_id is not None and len(gpus) > 0:
        gpu_percent = gpus[gpu_id].load
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024. / 1024.
    return round(cpu_mem, 4), round(gpu_mem, 4), round(gpus[gpu_id].memoryUtil * 100, 4)

class Predictor(object):
    """
    Args:
        model_dir (str): root path of model.pdiparams, model.pdmodel
        use_gpu (bool): whether use gpu
        batch_size (int): size of pre batch in inference
        threshold (float): threshold to reserve the result for output.
    """

    def __init__(self,
                 model_dir,
                 use_gpu=False,
                 batch_size=1,
                 cpu_threads=1,
                 enable_mkldnn=False):
        self.predictor, self.config = load_predictor(
            model_dir,
            batch_size=batch_size,
            use_gpu=use_gpu,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn)
        self.inference_time = 0.0

    def predict(self, warmup=0, repeats=1):
        results = None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            data = np.load("../dataset/EasyOCR_recognizer/img.npy")
            input_tensor.copy_from_cpu(data)
        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            output_tensor = self.predictor.get_output_handle(output_names[0])
            results = output_tensor.copy_to_cpu()
        start_time = time.time()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            output_tensor = self.predictor.get_output_handle(output_names[0])
            results = output_tensor.copy_to_cpu()
        end_time = time.time()
        self.inference_time = (end_time - start_time) / repeats
        return results

def load_predictor(model_dir,
                   batch_size=1,
                   use_gpu=False,
                   cpu_threads=1,
                   enable_mkldnn=False):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        use_gpu (bool): whether use gpu
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    """
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))
    if use_gpu:
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
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
    predictor = create_predictor(config)
    return predictor, config

def main():
    # for trace
    predictor = Predictor("pd_model_trace/inference_model/", 
                        use_gpu=True, 
                        cpu_threads=1, 
                        enable_mkldnn=False)
    predictor.predict(warmup=10, repeats=10)
    cm, gm, gu = get_current_memory_mb()
    cost_time = predictor.inference_time
    
    #record change
    if os.path.exists('result_mem_trace.txt'):
        with open('result_mem_trace.txt','r') as f1:
            lines = f1.readlines()
            inference_time_pre = lines[0].strip().split(',')[0].split(':')[1]
            cpu_mem_pre = lines[1].strip().split(',')[0].split(':')[1]
            gpu_mem_pre = lines[2].strip().split(',')[0].split(':')[1]
            gpu_percent_pre = lines[3].strip().split(',')[0].split(':')[1]

        inference_time_change = cost_time - float(inference_time_pre)
        cpu_mem_change = cm - float(cpu_mem_pre)
        gpu_mem_change = gm - float(gpu_mem_pre)
        gpu_percent_change = gu - float(gpu_percent_pre)
        if cpu_mem_change >= 1000 or gpu_mem_change >= 1000:
            assert 'change is so big! please check the model!'
        with open('result_mem_trace.txt','w') as f2:
            f2.write("inference_time:"+ str(cost_time)+ ",change:"+ str(inference_time_change)+ "\n")
            f2.write("cpu_mem:"+ str(cm)+ ",change:"+ str(cpu_mem_change)+ "\n")
            f2.write("gpu_mem:"+ str(gm)+ ",change:"+ str(gpu_mem_change)+ "\n")
            f2.write("gpu_percent:"+ str(gu)+ ",change:"+ str(gpu_percent_change)+ "\n")
        f1.close()
        f2.close()
    else:        
        with open('result_mem_trace.txt','w') as f1:
            f1.write("inference_time:"+ str(cost_time)+ ",change:0"+ "\n")
            f1.write("cpu_mem:"+ str(cm)+ ",change:0"+ "\n")
            f1.write("gpu_mem:"+ str(gm)+ ",change:0"+ '\n')
            f1.write("gpu_percent:"+ str(gu)+ ",change:0"+ '\n')
        f1.close()
        
    print_info = {
        'inference_time_trace': cost_time,
        'cpu_mem_trace': cm,
        'gpu_mem_trace': gm,
        'gpu_percent_trace': gu
    }
    return print_info
    

if __name__ == '__main__':
    paddle.enable_static()

    print_info = main()
    
