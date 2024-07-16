import onnx
import onnxruntime as rt
import os
import numpy as np
import sys
import pickle


model = onnx.load('../dataset/modelGen/modelGen_0256.onnx')
for i in range(4):
    for output in model.graph.output:
        model.graph.output.remove(output)
#for idx in range(0, len(check_names), -1):

# check_name = 'model/leaky_re_lu_70/LeakyRelu:0'
check_name = "ConvTranspose__264:0"
temp = onnx.helper.make_tensor_value_info(check_name, onnx.TensorProto.FLOAT, [-1])
model.graph.output.insert(0, temp)
print(model.graph.output)

with open('tmp.onnx', 'wb') as f:
    f.write(model.SerializeToString())
model_file = 'tmp.onnx'
sess = rt.InferenceSession(model_file)
inputs_dict = {}
# np.random.seed(6)

# input_data = np.load("input_yolov3.npy")
input_data = np.random.randn(1, 768, 512, 51).astype("float32")
inputs_dict[sess.get_inputs()[0].name] = input_data

 
import time
start = time.time()
repeats =  1
for i in range(repeats):
    result = sess.run(None, input_feed=inputs_dict)
    with open("onnx_temp.pkl", "wb") as fw:
        pickle.dump(result, fw)
    print(result[0].shape)
end = time.time()
