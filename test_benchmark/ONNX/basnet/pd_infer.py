import paddle
import paddle.fluid as fluid
import numpy as np
import pickle
import sys

f = open('result.txt', 'w')
f.write("======BASNET: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test Dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(
        dirname="pd_model_dygraph/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
    data = np.load("../dataset/basnet/input_0.npy")
    result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)

    result = np.array(result)
    with open("../dataset/basnet/result.pkl", "rb") as fr:
        onnx_result = pickle.load(fr)

    onnx_result = np.array(onnx_result)
    is_successd = True
    for i in range(8):
        diff = result[i] - onnx_result[i]
        max_abs_diff = np.fabs(diff).max()
        if max_abs_diff >= 1e-05:
            relative_diff_all = np.fabs(diff) / np.fabs(result[i])
            # relative_diff = max_abs_diff / np.fabs(onnx_result[i]).max()
            relative_diff = relative_diff_all.max()
            if relative_diff >= 1e-05:
                is_successd = False
    if is_successd:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!!Failed\n")
