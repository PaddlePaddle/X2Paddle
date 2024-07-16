# onnx simplifier
# python -m onnxsim GPEN-256.onnx GPEN-256_sim.onnx
# 进行转换
x2paddle -f onnx -m ../dataset/GPEN/GPEN-256_sim.onnx -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
