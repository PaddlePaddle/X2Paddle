import paddle
import numpy as np
import sys

f = open('result.txt', 'w')
f.write("======VGG11: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_dygraph/inference_model/model", executor=exe)
    data = np.load('../dataset/vgg11/input_0.npy')
    result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)
    paddle.disable_static()
    from pd_model_dygraph.x2paddle_code import main
    data = np.load('../dataset/vgg11/input_0.npy')
    data = paddle.to_tensor(data)
    result = main(data).numpy()

    onnx_result = np.load('../dataset/vgg11/result.npy')
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
