# 进行转换
x2paddle -f onnx -m ../dataset/Poselandmarks/pose_landmarks_saved_model_lite.onnx -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
