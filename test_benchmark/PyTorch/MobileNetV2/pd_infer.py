from __future__ import print_function
import paddle.fluid as fluid
import paddle
import sys
import os
import numpy
import pickle


def rel_err(x, y):
    return numpy.max(
        numpy.abs(x - y) / (numpy.maximum(numpy.abs(x), numpy.abs(y)) + 1e-08))


with open('../dataset/MobileNetV2/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)["data0"]
with open('../dataset/MobileNetV2/pytorch_output.pkl', 'rb') as oup:
    pytorch_output = pickle.load(oup)["output0"]
f = open("result.txt", "w")
f.write("======MobileNetV2: \n")
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
    df = pytorch_output - result[0]
    if numpy.max(numpy.fabs(df)) > 1e-04:
        print("Trace Failed", file=f)
    else:
        print("Trace Successed", file=f)

    # script
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs] = fluid.io.load_inference_model(
        dirname="pd_model_script/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
    result = exe.run(prog, feed={inputs[0]: input_data}, fetch_list=outputs)
    df = pytorch_output - result[0]
    if numpy.max(numpy.fabs(df)) > 1e-04:
        print("Script Failed", file=f)
    else:
        print("Script Successed", file=f)
except:
    print("!!!Failed\n", file=f)
f.close()
