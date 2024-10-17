import paddle
import numpy as np
import sys
import os

f = open('result.txt', 'w')
f.write("======MNasNet: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_dygraph/inference_model/model", executor=exe)
    data = np.load('../dataset/MNASNetA1/input.npy')
    result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)

    tf_result = np.load('../dataset/MNASNetA1/result.npy')
    diff = result[-1] - tf_result
    max_abs_diff = np.fabs(diff).max()

    if max_abs_diff < 1e-05:
        f.write("Dygraph Successed\n")
    else:
        print(max_abs_diff)
        relative_diff = max_abs_diff / np.fabs(tf_result).max()
        if relative_diff < 1e-05:
            f.write("Dygraph Successed\n")
        else:
            print(relative_diff)
            f.write("!!!!!Dygraph Failed\n")
except Exception as e:
    print(e)
    f.write("!!!!!Failed\n")
