import paddle
import paddle.fluid as fluid
import numpy as np
import pickle
import sys
import os

f = open('result.txt', 'w')
f.write("======UNet-unet: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    
    # test dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(dirname="pd_model_dygraph/inference_model/", 
                                                            executor=exe, 
                                                            model_filename="model.pdmodel",
                                                            params_filename="model.pdiparams")
    data = [np.load('../dataset/UNet-UNet/input.npy')]
    result = exe.run(prog, feed={inputs[0]:data[0]}, fetch_list=outputs)
   
    tf_results = np.load('../dataset/UNet-UNet/result.npy')

    abs_diff = list()
    if not isinstance(tf_results, list):
        tf_results = [tf_results]
    for i, r in enumerate(tf_results):
        abs_diff.append(np.fabs(result[i] - tf_results[i]))

    abs_pass = True
    for i in range(len(abs_diff)):
        if abs_diff[i].max() > 1e-04:
            abs_pass = False

    rel_pass = True
    if not abs_pass:
        for i in range(len(abs_diff)):
            if abs_diff[i].max() / np.fabs(tf_results[i]).max() > 1e-04:
                rel_pass = False

    if abs_pass or rel_pass:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
except Exception as e:
    print(e)
    f.write("!!!!!Failed\n")
