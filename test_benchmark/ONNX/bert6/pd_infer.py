import paddle
import paddle.fluid as fluid
import numpy as np
import pickle
import sys

with open('../dataset/bert6/inputs.pkl', 'rb') as inp:
    input_data = pickle.load(inp)
with open('../dataset/bert6/outputs.pkl', 'rb') as oup:
    onnx_result = pickle.load(oup)
onnx_result = list(onnx_result.values())
f = open('result.txt', 'w')
f.write("======bert6: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(
        dirname="pd_model_dygraph/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
    result = exe.run(prog,
                     feed={
                         inputs[0]: input_data["input_ids"],
                         inputs[1]: input_data["input_masks"]
                     },
                     fetch_list=outputs)

    is_successd = True
    for i in range(3):
        diff = result[i] - onnx_result[i]
        max_abs_diff = np.fabs(diff).max()
        if max_abs_diff >= 1e-05:
            relative_diff = max_abs_diff / np.fabs(onnx_result[i]).max()
            if relative_diff >= 1e-05:
                is_successd = False
    if is_successd:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")
