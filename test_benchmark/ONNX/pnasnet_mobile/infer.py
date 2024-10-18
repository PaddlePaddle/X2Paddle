import paddle
import numpy as np
import pickle
import sys

paddle.enable_static()
exe = paddle.static.Executor(paddle.CPUPlace())

# test dygraph
[prog, inputs, outputs] = paddle.static.load_inference_model(
    path_prefix="pd_model_dygraph/inference_model/model", executor=exe)

data = np.load('../dataset/pnasnet_mobile/input.npy')
result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)
