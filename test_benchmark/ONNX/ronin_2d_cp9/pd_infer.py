import paddle.fluid as fluid
import paddle
import numpy as np
import sys
import pickle

f = open('result.txt', 'w')
f.write("======ronin_2d: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs
     ] = fluid.io.load_inference_model(dirname="pd_model/inference_model/",
                                       executor=exe,
                                       model_filename="model.pdmodel",
                                       params_filename="model.pdiparams")
    data = np.load('../dataset/ronin_2d_cp9/input_ronin.npy')
    result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)

    with open("../dataset/ronin_2d_cp9/result_ronin_outputs.pkl", "rb") as fr:
        onnx_result = pickle.load(fr)
    is_successd = True
    for i in range(1):
        diff = result[i] - onnx_result[i]
        max_abs_diff = np.fabs(diff).max()
        if max_abs_diff >= 1e-05:
            relative_diff_all = max_abs_diff / np.fabs(onnx_result[i]).max()
            relative_diff = relative_diff_all.max()
            if relative_diff >= 1e-05:
                is_successd = False
    if is_successd:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")
