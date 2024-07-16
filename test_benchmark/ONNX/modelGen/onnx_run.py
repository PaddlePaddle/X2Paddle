import onnxruntime as rt
import os
import numpy as np
import sys
import pickle
 
model_dir = '../dataset/modelGen/modelGen_0256.onnx'
 
sess = rt.InferenceSession(model_dir)
 
inputs_dict = {}
input_list = list()
np.random.seed(6)

input_0 = np.random.randn(1, 768, 512, 51).astype("float32")

np.save('../dataset/modelGen/inputs.npy', input_0)
 
inputs_dict[sess.get_inputs()[0].name] = input_0

import time
start = time.time()
repeats =  1
for i in range(repeats):
    result = sess.run(None, input_feed=inputs_dict)
    with open("outputs.pkl", "wb") as fw:
        pickle.dump(result, fw)
    print(len(result))
    print(result[0].shape)

end = time.time()
