import paddle.fluid as fluid
import numpy as np
import sys

# model_name = sys.argv[1]
# model_dir = sys.argv[2]

# f = open('result.txt', 'w')
# f.write("======{}: ".format(model_name[:-1]))
# try:
#     exe = fluid.Executor(fluid.CPUPlace())
#     [prog, inputs, outputs] = fluid.io.load_inference_model(dirname=model_dir, executor=exe)
#     data = np.load('input_0.npy')
#     result = exe.run(prog, feed={inputs[0]:data}, fetch_list=outputs)

#     tf_result = np.load('result.npy')
#     diff = result[0] - tf_result
#     max_abs_diff = np.fabs(diff).max()

#     if max_abs_diff < 1e-05:
#         f.write("Successed\n")
#     else:
#         relative_diff = max_abs_diff / np.fabs(tf_result).max()
#         if relative_diff < 1e-05:
#             f.write("Successed\n")
#         else:
#             f.write("!!!!!Failed\n")
# except:
#     f.write("!!!!!Failed\n")
#     raise

    
import paddle.fluid as fluid
import paddle
import numpy as np
import sys


f = open('result.txt', 'w')
f.write("======Circledet: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(dirname="pd_model_dygraph/inference_model/", 
                                                            executor=exe, 
                                                            model_filename="model.pdmodel",
                                                            params_filename="model.pdiparams")
    data = np.load('../dataset/circledet/input_0.npy')
    result = exe.run(prog, feed={inputs[0]:data}, fetch_list=outputs)


    onnx_result = np.load('../dataset/circledet/result.npy')
    diff = result[0] - onnx_result
    max_abs_diff = np.fabs(diff).max()
    print(max_abs_diff)
    if max_abs_diff < 1e-05:
        f.write("Dygraph Successed\n")
    else:
        relative_diff = max_abs_diff / np.fabs(onnx_result).max()
        if relative_diff < 1e-05:
            f.write("Dygraph Successed\n")
        else:
            f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")
