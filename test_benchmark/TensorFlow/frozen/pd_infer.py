import paddle
import paddle.fluid as fluid
import numpy as np
import numpy
import sys
import pickle
import os

with open('../dataset/frozen/inputs.pkl', 'rb') as inp:
    input_data = pickle.load(inp)
tf_output = np.load("../dataset/frozen/outputs.npy")

f = open('result.txt', 'w')
f.write("======frozen: \n")
# try:
paddle.enable_static()
exe = paddle.static.Executor(paddle.CPUPlace())

# test dygraph
[prog, inputs, outputs
 ] = fluid.io.load_inference_model(dirname="pd_model_dygraph/inference_model/",
                                   executor=exe,
                                   model_filename="model.pdmodel",
                                   params_filename="model.pdiparams")
result = exe.run(prog,
                 feed={
                     inputs[0]: input_data["ipt0"],
                     inputs[1]: input_data["ipt1"],
                     inputs[2]: input_data["ipt2"]
                 },
                 fetch_list=outputs)
print(tf_output.shape)
print(result[0].shape)
df = tf_output - result[0]
print((numpy.max(numpy.fabs(df))))
if numpy.max(numpy.fabs(df)) > 2e-04:
    print("Dygraph Failed", file=f)
else:
    print("Dygraph Successed", file=f)
# except Exception as e:
#     f.write("!!!!!Failed")
#     sys.stderr.write("{}\n".format(e))
#     sys.exit(-1)
