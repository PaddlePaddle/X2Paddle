# 进行转换
x2paddle --framework=onnx --model=../dataset/SwinTransformer/jueyuanziyiwu_swin_tiny_byol_sim.onnx --save_dir=pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
