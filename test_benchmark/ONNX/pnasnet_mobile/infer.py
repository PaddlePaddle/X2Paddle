import paddle
import paddle.fluid as fluid
import numpy as np
import pickle
import sys

paddle.enable_static()
exe = paddle.static.Executor(paddle.CPUPlace())

# test dygraph
[prog, inputs, outputs
 ] = fluid.io.load_inference_model(dirname="pd_model_dygraph/inference_model/",
                                   executor=exe,
                                   model_filename="model.pdmodel",
                                   params_filename="model.pdiparams")
data = np.load('../dataset/pnasnet_mobile/input.npy')
result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)
