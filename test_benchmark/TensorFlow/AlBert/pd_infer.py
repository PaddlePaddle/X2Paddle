import paddle
import paddle.fluid as fluid
import numpy
import sys
import pickle
import os

with open('../dataset/AlBert/tf_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)
with open('../dataset/AlBert/tf_output.pkl', 'rb') as oup:
    tf_output = pickle.load(oup)

f = open('result.txt', 'w')
f.write("======AlBert: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(
         dirname="pd_model_dygraph/inference_model/",
         executor=exe,
         model_filename="model.pdmodel",
         params_filename="model.pdiparams")

    result = exe.run(inference_program,
                     feed={
                         feed_target_names[0]: input_data[0]["input_ids"],
                         feed_target_names[1]: input_data[0]["input_mask"],
                         feed_target_names[2]: input_data[0]["segment_ids"]
                     },
                     fetch_list=fetch_targets)

    df = tf_output[0] - result
    if numpy.max(numpy.fabs(df)) > 1e-04:
        print("Dygraph Failed", file=f)
    else:
        print("Dygraph Successed", file=f)
except Exception as e:
    f.write("!!!!!Failed\n")
    sys.stderr.write("{}\n".format(e))
    sys.exit(-1)
