# 进行转换
echo "-1,3,1280,1280" | x2paddle -f onnx -m ../dataset/Seg/seg.onnx -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
