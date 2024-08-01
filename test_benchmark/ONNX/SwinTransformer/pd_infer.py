import paddle.fluid as fluid
import paddle
import numpy as np
import sys
import pickle

f = open('result.txt', 'w')
f.write("======SwinTransformer: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(
        dirname="pd_model_dygraph/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
    LN_nums = 0
    for i, op in enumerate(prog.block(0).ops):
        if op.type in ['feed', 'fetch']:
            continue
        if op.type == 'layer_norm':
            LN_nums += 1
    input_data = np.load("../dataset/SwinTransformer/input_swin.npy")
    result = exe.run(prog, feed={inputs[0]: input_data}, fetch_list=outputs)

    with open("../dataset/SwinTransformer/result_swin_outputs.pkl", "rb") as fr:
        onnx_result = pickle.load(fr)

    is_successd = True
    ## compare LN nums
    if LN_nums != 32:
        is_successd = False
    for i in range(2):
        diff = result[i][:20] - onnx_result[i][:20]
        max_abs_diff = np.fabs(diff).max()
        if max_abs_diff >= 1e-05:
            relative_diff_all = max_abs_diff / np.fabs(onnx_result[i]).max()
            relative_diff = relative_diff_all.max()
            if relative_diff >= 1e-05:
                is_successd = False
    if is_successd:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")
