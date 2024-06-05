from __future__ import print_function
import paddle
import paddle.fluid as fluid
import sys
import os
import numpy as np
import pickle


f = open("result.txt", "w")
f.write("======MTCNN1: \n")

try:
    with open('caffe_input.pkl', 'rb') as inp:
        input_data = pickle.load(inp)["data0"]
    with open('caffe_output.pkl', 'rb') as oup:
        caffe_result = pickle.load(oup)["prob1"]
    
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
            
    # test dygraph
    [prog, 
     feed_target_names, 
     fetch_targets] = fluid.io.load_inference_model(dirname="pd_model_dygraph/inference_model/", 
                                                    executor=exe, 
                                                    model_filename="model.pdmodel",
                                                    params_filename="model.pdiparams")
    result = exe.run(prog, feed={feed_target_names[0]:input_data}, fetch_list=fetch_targets)
    diff = caffe_result[0] - np.asarray(result[-1][0])
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

