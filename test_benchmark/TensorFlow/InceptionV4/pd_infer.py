import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import os

f = open('result.txt', 'w')
f.write("======InceptionV4: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
            
    # test dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(dirname="pd_model_dygraph/inference_model/", 
                                                            executor=exe, 
                                                            model_filename="model.pdmodel",
                                                            params_filename="model.pdiparams")
    data = np.load('../dataset/InceptionV4/input.npy')
    result = exe.run(prog, feed={inputs[0]:data}, fetch_list=outputs)
    
    tf_result = np.load('../dataset/InceptionV4/result.npy')
    diff = result[0] - tf_result
    max_abs_diff = np.fabs(diff).max()
    
    if max_abs_diff < 1e-05:
        f.write("Dygraph Successed\n")
    else:
        relative_diff = max_abs_diff / np.fabs(tf_result).max()
        if relative_diff < 1e-05:
            f.write("Dygraph Successed\n")
        else:
            print('----', relative_diff)
            f.write("!!!!!Dygraph Failed\n")
except Exception as e:
    f.write("!!!!!Failed\n")
    sys.stderr.write("{}\n".format(e))
    sys.exit(-1)
