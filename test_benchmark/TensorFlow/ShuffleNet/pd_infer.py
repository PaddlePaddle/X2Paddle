import paddle
import numpy as np
import sys
import os

f = open('result.txt', 'w')
f.write("======ShuffleNet: \n")
data = np.load("../dataset/ShuffleNet/input.npy")
tf_result = np.load("../dataset/ShuffleNet/result.npy")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_dygraph/inference_model/model", executor=exe)
    result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)

    diff = result[0] - tf_result
    max_abs_diff = np.fabs(diff).max()

    if max_abs_diff < 1e-05:
        f.write("Dygraph Successed\n")
    else:
        relative_diff = max_abs_diff / np.fabs(tf_result).max()
        if relative_diff < 1e-05:
            f.write("Dygraph Successed\n")
        else:
            f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")
