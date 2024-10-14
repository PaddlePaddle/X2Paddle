from __future__ import print_function
import paddle
import sys
import os
import numpy as np
import pickle

f = open("result.txt", "w")
f.write("======inceptionv3: \n")

try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    data = np.random.rand(1, 224, 224, 3).astype("float32")

    # test dygrah
    [inference_program, feed_target_names,
     fetch_targets] = paddle.static.load_inference_model(
         path_prefix="pd_model_dygraph/inference_model/model", executor=exe)

    result = exe.run(inference_program,
                     feed={feed_target_names[0]: data},
                     fetch_list=fetch_targets)
    f.write("Dygraph Successed\n")
except:
    f.write("!!!!!Failed\n")
f.close()
