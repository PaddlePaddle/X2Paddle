from __future__ import print_function
import paddle
import sys
import os
import numpy as np
import pickle

f = open("result.txt", "w")
f.write("======AlexNet: \n")

try:
    with open('../dataset/AlexNet/caffe_input.pkl', 'rb') as inp:
        input_data = pickle.load(inp)["data0"]
    with open('../dataset/AlexNet/caffe_output.pkl', 'rb') as oup:
        caffe_result = pickle.load(oup)["prob"]

    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    # test dygraph
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_dygraph/inference_model/model", executor=exe)
    # test dygraph
    paddle.disable_static()
    from pd_model_dygraph.x2paddle_code import main
    input_data = paddle.to_tensor(input_data)
    result = main(input_data)
    diff = caffe_result[0] - result[0].numpy()
    max_abs_diff = np.fabs(diff).max()
    if max_abs_diff < 1e-05:
        f.write("Dygraph Successed\n")
    else:
        relative_diff = max_abs_diff / np.fabs(caffe_result).max()
        if relative_diff < 1e-05:
            f.write("Dygraph Successed\n")
        else:
            f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")

f.close()
