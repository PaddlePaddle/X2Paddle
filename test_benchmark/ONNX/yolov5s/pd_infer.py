import paddle.fluid as fluid
import paddle
import numpy as np
import sys

f = open('result.txt', 'w')
f.write("======YOLOv5s: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(
        dirname="pd_model_dygraph/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
    data = np.load('../dataset/yolov5s/input_0.npy')
    result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)

    paddle.disable_static()
    from pd_model_dygraph.x2paddle_code import main
    data = np.load('../dataset/yolov5s/input_0.npy')
    data = paddle.to_tensor(data)
    result = main(data)[0].numpy()

    onnx_result = np.load('../dataset/yolov5s/result.npy')
    diff = result[0] - onnx_result
    max_abs_diff = np.fabs(diff).max()

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
