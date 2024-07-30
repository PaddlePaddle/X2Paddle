import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import os


def check_fuser(file_path, standard_code):
    with open(file_path, "r") as fr:
        code = fr.readlines()
        code = "    ".join(code)
        if standard_code in code:
            return True
        else:
            return False


f = open('result.txt', 'w')
f.write("======EfficientNet: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(
        dirname="pd_model_dygraph/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
    data = np.load('../dataset/EfficientNet/input.npy')
    result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)

    tf_result = np.load('../dataset/EfficientNet/result.npy')
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
    if not check_fuser("pd_model_dygraph/x2paddle_code.py",
                       "paddle.nn.BatchNorm"):
        f.write("!!!!!Dygraph BatchNorm Fuser Failed\n")
except Exception as e:
    f.write("!!!!!Failed\n")
