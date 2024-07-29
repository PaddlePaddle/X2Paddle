import onnxruntime as rt
import os
import numpy as np
inputs_dict = {}
onnx_file = None
input_file = None
path = './'
for name in os.listdir(path):
    if 'onnx' in name:
        onnx_file = os.path.join(path, name)
    if 'input_0.npy' == name:
        input_file = os.path.join(path, name)
sess = rt.InferenceSession(onnx_file)
inputs_dict[sess.get_inputs()[0].name] = np.load(input_file)
result_path = os.path.join(path, 'result.npy')
res_onnx = sess.run(None, input_feed=inputs_dict)
print(res_onnx[0])
np.save(result_path, res_onnx[0])
