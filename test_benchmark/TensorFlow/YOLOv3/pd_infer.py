import paddle
import numpy as np
import pickle
import sys
import os

f = open('result.txt', 'w')
f.write("======YOLOv3: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_dygraph/inference_model/model", executor=exe)
    data = [np.load('../dataset/YOLOv3/input.npy')]
    result = exe.run(prog, feed={inputs[0]: data[0]}, fetch_list=outputs)

    with open('../dataset/YOLOv3/result.pkl', 'rb') as f1:
        tf_results = pickle.load(f1)

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
