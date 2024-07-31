from __future__ import print_function
import paddle.fluid as fluid
import paddle
import sys
import os
import numpy as np
import pickle

img = np.load("../dataset/EasyOCR_detector/img.npy")
with open("../dataset/EasyOCR_detector/result.pkl", "rb") as fr:
    pytorch_output = pickle.load(fr)
f = open("result.txt", "w")
f.write("======EasyOCR detector: \n")
try:
    # trace
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs] = fluid.io.load_inference_model(
        dirname="pd_model_trace/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
    result = exe.run(prog, feed={inputs[0]: img}, fetch_list=outputs)
    df0 = pytorch_output["output0"] - result[0]
    df1 = pytorch_output["output1"] - result[1]
    if np.max(np.fabs(df0)) > 1e-04 or np.max(np.fabs(df1)) > 1e-04:
        print("Trace Failed", file=f)
    else:
        print("Trace Successed", file=f)

except:
    print("!!!!!Failed", file=f)
f.close()
