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


with open('../dataset/BertModel_SpanBert/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)
with open('../dataset/BertModel_SpanBert/pytorch_output.pkl', 'rb') as oup:
    pytorch_output = pickle.load(oup)

f = open("result.txt", "w")
f.write("======BertModel_SpanBert: \n")
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
                         inputs[1]: input_data["attention_mask"],
                         inputs[2]: input_data["token_type_ids"]
                     },
                     fetch_list=outputs)
    df = pytorch_output["output0"] - result[0]
    print(numpy.max(numpy.fabs(df)))
    if numpy.max(numpy.fabs(df)) > 1e-04:
        print("Dygraph Failed\n", file=f)
    else:
        print("Dygraph Successed\n", file=f)
except:
    print("!!!Failed\n", file=f)
f.close()
