import paddle
import paddle.fluid as fluid
import numpy as np
import pickle
import sys

f = open('result.txt', 'w')
f.write("======mmdetection_RetinaNet: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    # test Dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(dirname="pd_model_dygraph/inference_model/",
                                                            executor=exe,
                                                            model_filename="model.pdmodel",
                                                            params_filename="model.pdiparams")
    data = np.load("../dataset/mmdetection_retina/real_img_data_retina_50.npy")
    result = exe.run(prog, feed={inputs[0]:data}, fetch_list=outputs)

    with open("../dataset/mmdetection_retina/result_retina_onnx.pkl", "rb") as fr:
        onnx_result = pickle.load(fr)

    is_successd = True
    for i in range(2):
        diff = result[i] - onnx_result[i]
        max_abs_diff = np.fabs(diff).max()
        if max_abs_diff >= 1e-05:
            relative_diff = max_abs_diff / np.fabs(onnx_result[i]).max()
            if relative_diff >= 1e-05:
                is_successd = False
    if is_successd:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")
