import paddle
import sys
import os
import numpy as np

img = np.load("../dataset/yolov5/input.npy")
pytorch_output = np.load("../dataset/yolov5/output.npy")
f = open("result.txt", "w")
f.write("======yolov5 recognizer: \n")
try:
    # trace
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_trace/inference_model/model", executor=exe)
    result = exe.run(prog, feed={inputs[0]: img}, fetch_list=outputs)
    df = pytorch_output - result[0][0]
    if np.max(np.fabs(df)) > 3e-03:
        print("Trace Failed", file=f)
    else:
        print("Trace Successed", file=f)

except:
    print("!!!!!Failed", file=f)

f.close()
