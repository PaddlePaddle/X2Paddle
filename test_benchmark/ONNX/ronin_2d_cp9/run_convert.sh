# 进行转换
x2paddle -f onnx -m ../dataset/ronin_2d_cp9/ronin_2d_cp9.onnx -s pd_model
# 运行推理程序
python pd_infer.py
