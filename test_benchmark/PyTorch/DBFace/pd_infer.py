from __future__ import print_function

import paddle
import common
import numpy as np
import cv2
import pickle
import os

with open('../dataset/DBFace/outputs.pkl', 'rb') as oup:
    pytorch_output = pickle.load(oup)
file = "../dataset/DBFace/datas/selfie.jpg"
image = common.imread(file)
mean = [0.408, 0.447, 0.47]
std = [0.289, 0.274, 0.278]
image = common.pad(image)
image = ((image / 255.0 - mean) / std).astype(np.float32)
image = np.expand_dims(image.transpose(2, 0, 1), axis=0)
print(image.shape)
f = open("result.txt", "w")
f.write("======DBFace: \n")
try:
    # trace
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_trace/inference_model/model", executor=exe)
    result = exe.run(prog, feed={inputs[0]: image}, fetch_list=outputs)

    df0 = pytorch_output["hm"] - result[0]
    df1 = pytorch_output["box"] - result[1]
    df2 = pytorch_output["landmark"] - result[2]
    if np.max(np.fabs(df0)) > 1e-04 or np.max(np.fabs(df1)) > 1e-04 or np.max(
            np.fabs(df2)) > 1e-04:
        print("Trace Failed", file=f)
    else:
        print("Trace Successed", file=f)
except:
    print("!!!!!Failed", file=f)
f.close()
