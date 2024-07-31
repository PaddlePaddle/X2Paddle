import paddle
import paddle.fluid as fluid
import numpy as np
import pickle
import numpy
import sys
import os

with open('../dataset/KerasBert/tf_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)
with open('../dataset/KerasBert/tf_output.pkl', 'rb') as oup:
    tf_output = pickle.load(oup)

f = open('result.txt', 'w')
f.write("======KerasBert: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(
        dirname="pd_model_dygraph/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
    result = exe.run(prog,
                     feed={
                         inputs[0]: input_data[0]["input_ids"],
                         inputs[1]: input_data[0]["segment_ids"]
                     },
                     fetch_list=outputs)

    df = tf_output[0] - result
    if numpy.max(numpy.fabs(df)) > 1e-04:
        print("Dygraph Failed", file=f)
    else:
        print("Dygraph Successed", file=f)
except Exception as e:
    f.write("!!!!!Failed\n")
    sys.stderr.write("{}\n".format(e))
    sys.exit(-1)

# import paddle
# import paddle.fluid as fluid
# import numpy as np
# import pickle
# import numpy
# import sys

# with open('tf_input.pkl', 'rb') as inp:
#     input_data = pickle.load(inp)
# with open('tf_output.pkl', 'rb') as oup:
#     tf_output = pickle.load(oup)

# print(len(input_data))
# f = open('result.txt', 'w')
# f.write("======KerasBert: \n")
# try:
#     paddle.enable_static()
#     exe = paddle.static.Executor(paddle.CPUPlace())

#     # test static
#     [prog, inputs, outputs] = fluid.io.load_inference_model(dirname="pd_model_static/inference_model",
#                                                             executor=exe)
#     for i in range(10):
#         result = exe.run(prog,
#                          feed={inputs[0]: input_data[i]["input_ids"],
#                                inputs[1]: input_data[i]["segment_ids"]},
#                          fetch_list=outputs)
#         print(result)
# except Exception as e:
#     f.write("!!!!!Failed")
#     sys.stderr.write("{}\n".format(e))
#     sys.exit(-1)
