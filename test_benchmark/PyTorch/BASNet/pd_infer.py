from __future__ import print_function
import paddle.fluid as fluid
import paddle
import sys
import os
import numpy
import pickle

with open('../dataset/BASNet/input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)["input"]
with open('../dataset/BASNet/output.pkl', 'rb') as oup:
    pytorch_output = pickle.load(oup)
f = open("result.txt", "w")
f.write("======BASNet: \n")
try:
    # trace
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs] = fluid.io.load_inference_model(
        dirname="pd_model_trace/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
    result = exe.run(prog, feed={inputs[0]: input_data}, fetch_list=outputs)
    result_string = "Trace Successed"
    for i in range(len(result)):
        df = pytorch_output[i].detach().numpy() - result[i]
        if numpy.max(numpy.fabs(df)) > 1e-04:
            result_string = "Trace Failed"
    print(result_string, file=f)

except:
    print("!!!!!Failed", file=f)
f.close()
