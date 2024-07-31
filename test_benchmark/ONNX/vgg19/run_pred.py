import onnxruntime as rt
import os
import numpy as np
import sys
import pickle

model_dir = 'vgg19_based.onnx'

sess = rt.InferenceSession(model_dir)

inputs_dict = {}
np.random.seed(6)
input_0 = np.random.rand(1, 1, 160, 101).astype('float32')
np.save('./input.npy', input_0)

inputs_dict[sess.get_inputs()[0].name] = input_0

import time

start = time.time()
repeats = 1
for i in range(repeats):
    result = sess.run(None, input_feed=inputs_dict)
    with open("output.pkl", "wb") as fw:
        pickle.dump(result, fw)
    print(result)
end = time.time()
