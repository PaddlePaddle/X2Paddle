import paddle

import numpy as np
import pickle
import sys
import os

f = open('result.txt', 'w')
f.write("======ResNet18_2: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    # test trace
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_trace/inference_model/model", executor=exe)
    data = np.load('../dataset/ResNet18_2/input.npy')
    result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)

    with open("../dataset/ResNet18_2/output.pkl", "rb") as fr:
        pytorch_result = pickle.load(fr)
    pytorch_result = list(pytorch_result.values())
    is_successd = True
    for i in range(3):
        diff = result[i] - pytorch_result[i]
        max_abs_diff = np.fabs(diff).max()
        if max_abs_diff >= 1e-05:
            relative_diff = max_abs_diff / np.fabs(pytorch_result[i]).max()
            if relative_diff >= 1e-05:
                is_successd = False
    if is_successd:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")
