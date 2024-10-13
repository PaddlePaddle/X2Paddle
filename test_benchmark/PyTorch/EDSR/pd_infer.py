from __future__ import print_function

import paddle
import sys
import os
import numpy as np

img = np.load("../dataset/EDSR/input.npy")
pytorch_output = np.load("../dataset/EDSR/result.npy")
f = open("result.txt", "w")
f.write("======EDSR recognizer: \n")
try:
    # trace
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_trace/inference_model/model", executor=exe)
    result = exe.run(prog, feed={inputs[0]: img}, fetch_list=outputs)
    df = pytorch_output - result
    if np.max(np.fabs(df)) > 1e-03:
        print("Trace Failed", file=f)
    else:
        print("Trace Successed", file=f)

    # script
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_script/inference_model/model", executor=exe)
    result = exe.run(prog, feed={inputs[0]: img}, fetch_list=outputs)
    df = pytorch_output - result
    if np.max(np.fabs(df)) > 1e-03:
        print("Script Failed", file=f)
    else:
        print("Script Successed", file=f)

except:
    print("!!!!!Failed", file=f)
f.close()
