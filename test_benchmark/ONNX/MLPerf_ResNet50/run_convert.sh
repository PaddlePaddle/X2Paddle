# 进行转换
echo "N" | x2paddle -f onnx -m ../dataset/MLPerf_ResNet50/resnet50_v1.onnx -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
