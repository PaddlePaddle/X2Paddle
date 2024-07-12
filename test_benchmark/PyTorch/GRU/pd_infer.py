from __future__ import print_function
import paddle.fluid as fluid
import paddle
import sys
import os
import numpy
import numpy as np

input_data = np.load("../dataset/GRU/input.npy")
pytorch_result = np.load("../dataset/GRU/output.npy")
f = open("result.txt", "w")
f.write("======gru: \n")
try:
    # trace
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs] = fluid.io.load_inference_model(dirname="pd_model_trace/inference_model/", 
					    executor=exe, 
					    model_filename="model.pdmodel",
					    params_filename="model.pdiparams")
    result = exe.run(prog, feed={inputs[0]:input_data}, fetch_list=outputs)
    abs_diff = np.max(np.abs(pytorch_result- result[0]))
    print(abs_diff)
    print("Trace Successed", file=f)
        
except:
    print("!!!!!Failed", file=f)
f.close()

