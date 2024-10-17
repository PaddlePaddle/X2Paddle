import paddle
import numpy as np
import sys
import pickle
import gzip

f = open('result.txt', 'w')
f.write("======YOLOv5s: \n")

paddle.enable_static()
exe = paddle.static.Executor(paddle.CPUPlace())

# test dygraph
[prog, inputs, outputs] = paddle.static.load_inference_model(
    path_prefix="pd_model_dygraph/inference_model/model", executor=exe)
# data = np.load('onnx_inputs.npy',allow_pickle=True)
with open("../dataset/yolov5s_fix_resize/onnx_inputs.pkl", "rb") as fr:
    data = pickle.load(fr)
    # print(data)
result = exe.run(prog, feed={inputs[0]: data["images"]}, fetch_list=outputs)

# paddle.disable_static()
# from pd_model_dygraph.x2paddle_code import main
# # data = np.load('onnx_inputs.pkl',allow_pickle=True)
# data = paddle.to_tensor(data)
# result = main(data)[0].numpy()

with open("../dataset/yolov5s_fix_resize/onnx_outputs.pkl", "rb") as fr:
    onnx_result = pickle.load(fr)
# onnx_result=pickle.load(onnx_result_1)
# onnx_result = np.load('onnx_putputs.npy',allow_pickle=True)
diff = result[0] - onnx_result
max_abs_diff = np.fabs(diff).max()

if max_abs_diff < 1e-05:
    f.write("Dygraph Successed\n")

else:
    relative_diff = max_abs_diff / np.fabs(onnx_result).max()
    print(relative_diff)
    if relative_diff < 1e-05:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
