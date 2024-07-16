# 进行转换
echo "N" | x2paddle -f onnx -m ../dataset/yolov5s/model.onnx -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
