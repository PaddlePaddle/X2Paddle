from __future__ import print_function
import paddle.fluid as fluid
import paddle
import sys
import os
import numpy
import pickle

input_data = numpy.load("../dataset/SwinTransformer/input.npy")
pytorch_output = numpy.load("../dataset/SwinTransformer/output.npy")
f = open("result.txt", "w")
f.write("======SwinTransformer: \n")
try:
    # trace
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs] = fluid.io.load_inference_model(dirname="pd_model_trace/inference_model/", 
                                                            executor=exe, 
                                                            model_filename="model.pdmodel",
                                                            params_filename="model.pdiparams")
    result = exe.run(prog, feed={inputs[0]:input_data}, fetch_list=outputs)
    df = pytorch_output - result
    if numpy.max(numpy.fabs(df)) > 1e-04:
        print("Trace Failed", file=f)
    else:
        print("Trace Successed", file=f)
except:
    print("!!!!!Failed", file=f)
f.close()
