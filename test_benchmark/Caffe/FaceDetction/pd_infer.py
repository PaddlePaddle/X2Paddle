from __future__ import print_function
import paddle
import paddle.fluid as fluid
import sys
import os
import numpy as np
import pickle

f = open("result.txt", "w")
f.write("======FaceDetection: \n")

try:
    input_data = np.random.rand(2, 3, 240, 320).astype("float32")
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    paddle.enable_static()
    [prog, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
        dirname="pd_model_dygraph/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")

    result = exe.run(prog,
                     feed={feed_target_names[0]: input_data},
                     fetch_list=fetch_targets,
                     return_numpy=False)
    f.write("Dygraph Successed\n")
except:
    f.write("!!!!!Failed\n")

f.close()
