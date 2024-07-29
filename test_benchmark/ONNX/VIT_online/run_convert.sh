# 进行转换
x2paddle --framework=onnx --model=../dataset/VIT_online/vit_online_onnxsim.onnx --save_dir=pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
