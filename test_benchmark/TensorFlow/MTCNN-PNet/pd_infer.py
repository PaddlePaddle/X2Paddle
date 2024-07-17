import paddle
import paddle.fluid as fluid
import numpy as np
import pickle
import sys
import os

def check_fuser(file_path, standard_code):
    with open(file_path, "r") as fr:
        code = fr.readlines()
        code = "    ".join(code)
        if standard_code in code:
            return True
        else:
            return False

f = open('result.txt', 'w')
f.write("======MTCNN-PNet: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
        
    # test dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(dirname="pd_model_dygraph/inference_model/", 
                                                            executor=exe, 
                                                            model_filename="model.pdmodel",
                                                            params_filename="model.pdiparams")
    data = np.load('../dataset/MTCNN-PNet/input.npy')
    result = exe.run(prog, feed={inputs[0]:data}, fetch_list=outputs)
   
    with open('../dataset/MTCNN-PNet/result.pkl', 'rb') as f1:
        tf_results = pickle.load(f1)

    abs_diff = list()
    if not isinstance(tf_results, list):
        tf_results = [tf_results]
    for i, r in enumerate(tf_results):
        abs_diff.append(np.fabs(result[i] - tf_results[i]))

    abs_pass = True
    for i in range(len(abs_diff)):
        if abs_diff[i].max() > 1e-05:
            abs_pass = False

    rel_pass = True
    if not abs_pass:
        for i in range(len(abs_diff)):
            if abs_diff[i].max() / np.fabs(tf_results[i]).max() > 1e-05:
                rel_pass = False

    if abs_pass or rel_pass:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
    if not check_fuser("pd_model_dygraph/x2paddle_code.py", "paddle.nn.PReLU"):
        f.write("!!!!!Dygraph PRelu Fuser Failed\n")
except Exception as e:
    print(e)
    f.write("!!!!!Failed\n")
