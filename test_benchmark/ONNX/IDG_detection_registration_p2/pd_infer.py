import paddle
import numpy as np
import pickle
import sys

f = open('result.txt', 'w')
f.write("======detection_registration_p2: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_dygraph/inference_model/model", executor=exe)
    with open(
            "../dataset/IDG_detection_registration_p2/detection_registration_p2_inputs.pkl",
            "rb") as fr:
        inputs_list = pickle.load(fr)
    result = exe.run(prog,
                     feed={
                         inputs[0]: inputs_list[0],
                         inputs[1]: inputs_list[1],
                         inputs[2]: inputs_list[2],
                         inputs[3]: inputs_list[3]
                     },
                     fetch_list=outputs)

    with open(
            "../dataset/IDG_detection_registration_p2/detection_registration_p2_outputs.pkl",
            "rb") as fr:
        onnx_result = pickle.load(fr)
    is_successd = True
    for i in range(3):
        diff = result[i] - onnx_result[i]
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
