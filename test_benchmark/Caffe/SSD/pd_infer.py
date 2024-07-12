from __future__ import print_function
import paddle
import paddle.fluid as fluid
import sys
import os
import numpy as np
import pickle


f = open("result.txt", "w")
f.write("======SSD: \n")

try:
    with open('../dataset/SSD/caffe_input.pkl', 'rb') as inp:
        input_data = pickle.load(inp)["data0"]
    with open('../dataset/SSD/caffe_output.pkl', 'rb') as oup:
        caffe_output = pickle.load(oup)
    
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    # test dygraph
    [prog, 
     feed_target_names, 
     fetch_targets] = fluid.io.load_inference_model(dirname="pd_model_dygraph/inference_model/", 
                                                    executor=exe, 
                                                    model_filename="model.pdmodel",
                                                    params_filename="model.pdiparams")
    result = exe.run(prog, feed={feed_target_names[0]:input_data}, fetch_list=fetch_targets, return_numpy=False)
    lod = result[0].lod()[0]
    res_pd =  np.asarray(result[0])
    res_cf = caffe_output['detection_out'][0]
    diffTest = True
    for i in range(len(lod)-1):
        count = lod[i+1]-lod[i]
        if count == 0:
            cf_need = res_cf[:, :, 0]
            if float(i) in cf_need:
                print("diff fail", file=f)
        else:
            a = res_pd[lod[i]: lod[i+1]]
            s = res_pd.shape[0]
            b = np.ones((s, 1))
            b *= i
            pres = np.concatenate((b, a),axis=1)
            diff = 1
            for i1 in range(res_cf.shape[0]):
                for i2 in range(res_cf.shape[1]):
                    cres = res_cf[i1, i2, :]
                    for ii in range(pres.shape[0]):
                        if diff > np.max(np.abs(cres - pres[ii, :])):
                            diff = np.max(np.abs(cres - pres[ii, :]))
            if diff > 1e-05:
                diffTest = False
                f.write("!!!!!Dygraph Failed\n")
                break
    if diffTest:
        f.write("Dygraph Successed\n")
except:
    f.write("!!!!!Failed\n")
f.close()

