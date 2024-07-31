import onnx
import onnxruntime

model = onnx.load('vgg19_based.onnx')
for i in range(5):
    for output in model.graph.output:
        model.graph.output.remove(output)
#for idx in range(0, len(check_names), -1):

check_name = '184'
temp = onnx.helper.make_tensor_value_info(check_name, onnx.TensorProto.FLOAT,
                                          [-1, -1])
model.graph.output.insert(0, temp)

with open('tmp.onnx', 'wb') as f:
    f.write(model.SerializeToString())
model_file = 'tmp.onnx'
sess = onnxruntime.InferenceSession(model_file)
