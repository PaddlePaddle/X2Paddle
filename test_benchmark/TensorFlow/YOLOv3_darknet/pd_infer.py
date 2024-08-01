import paddle
import paddle.fluid as fluid
import numpy as np
import pickle
import sys
import os

data = np.load('../dataset/YOLOv3_darknet/input.npy')
tf_result = np.load('../dataset/YOLOv3_darknet/output.npy')
f = open('result.txt', 'w')
f.write("======YOLOv3_darknet: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(
        dirname="pd_model_dygraph/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
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
            print('----', relative_diff)
            f.write("!!!!!Dygraph Failed\n")

except Exception as e:
    print(e)
    f.write("!!!!!Failed\n")
