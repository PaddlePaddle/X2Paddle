import paddle
import numpy as np
import pickle
import sys


f = open('result.txt', 'w')
f.write("======mobilestereonet: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = paddle.static.load_inference_model(path_prefix="pd_model/inference_model", 
                                                            executor=exe, 
                                                            model_filename="model.pdmodel",
                                                            params_filename="model.pdiparams")
    dummy_input_left = np.load("../dataset/Mobilestereonet/input_left.npy")
    dummy_input_right = np.load("../dataset/Mobilestereonet/input_right.npy")
    result = exe.run(prog, feed={inputs[0]:dummy_input_left, inputs[1]:dummy_input_right}, fetch_list=outputs)

    pytorch_result = np.load("../dataset/Mobilestereonet/result.npy")
    is_successd = True
    for i in range(1):
        diff = result[i] - pytorch_result
        max_abs_diff = np.fabs(diff).max()
        if max_abs_diff >= 1e-05:
            # relative_diff_all = np.fabs(diff) / np.fabs(result[i])
            relative_diff_all = max_abs_diff / np.fabs(pytorch_result).max()
            relative_diff = relative_diff_all.max()
            if relative_diff >= 1e-05:
                is_successd = False
    if is_successd:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")
