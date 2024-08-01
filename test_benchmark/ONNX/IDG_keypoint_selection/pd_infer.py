import paddle
import paddle.fluid as fluid
import numpy as np
import pickle
import sys

f = open('result.txt', 'w')
f.write("======keypoint_selection: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = fluid.io.load_inference_model(
        dirname="./pd_model_dygraph/inference_model/",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
    with open("../dataset/IDG_keypoint_selection/keypoint_selection_inputs.pkl",
              "rb") as fr:
        inputs_list = pickle.load(fr)

    feed_dict = dict()
    for i in range(12):
        feed_dict[inputs[i]] = inputs_list[i]
    result = exe.run(prog, feed=feed_dict, fetch_list=outputs)

    with open(
            "../dataset/IDG_keypoint_selection/keypoint_selection_outputs.pkl",
            "rb") as fr:
        onnx_result = pickle.load(fr)
    is_successd = True
    for i in range(2):
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
