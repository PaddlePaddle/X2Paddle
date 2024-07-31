from __future__ import print_function
import paddle.fluid as fluid
import paddle
import sys
import os
import numpy
import pickle

with open('../dataset/CamembertForQuestionAnswering/pytorch_input.pkl',
          'rb') as inp:
    input_data = pickle.load(inp)
with open('../dataset/CamembertForQuestionAnswering/pytorch_output.pkl',
          'rb') as oup:
    pytorch_output = pickle.load(oup)

f = open("result.txt", "w")
f.write("======CamembertForQuestionAnswering: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [prog, inputs, outputs
     ] = fluid.io.load_inference_model(dirname="pd_model/inference_model/",
                                       executor=exe,
                                       model_filename="model.pdmodel",
                                       params_filename="model.pdiparams")
    result = exe.run(prog,
                     feed={
                         inputs[0]: input_data["input_ids"],
                         inputs[1]: input_data["attention_mask"]
                     },
                     fetch_list=outputs)
    df = pytorch_output["output0"] - result[0]
    if numpy.max(numpy.fabs(df)) > 1e-04:
        print("Dygraph Failed\n", file=f)
    else:
        print("Dygraph Successed\n", file=f)
except:
    print("!!!Failed\n", file=f)
f.close()
