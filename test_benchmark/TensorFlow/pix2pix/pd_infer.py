import paddle.fluid as fluid
import paddle
import numpy as np
import sys
import pickle

f = open('result.txt', 'w')
f.write("======pix2pix: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs
     ] = fluid.io.load_inference_model(dirname="pd_model/inference_model/",
                                       executor=exe,
                                       model_filename="model.pdmodel",
                                       params_filename="model.pdiparams")
    data = np.load('../dataset/pix2pix/input_tensorflow_0120.npy')
    result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)

    with open("../dataset/pix2pix/result_tensorflow_0120.pkl", 'rb') as fr:
        tf_result = pickle.load(fr)
    diff = result[0] - tf_result[0]
    max_abs_diff = np.fabs(diff).max()

    if max_abs_diff < 1e-05:
        f.write("Dygraph Successed\n")
    else:
        relative_diff = max_abs_diff / np.fabs(tf_result[0]).max()
        if relative_diff < 1e-05:
            f.write("Dygraph Successed\n")
        else:
            f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")
