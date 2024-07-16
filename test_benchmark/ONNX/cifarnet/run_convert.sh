# 进行转换
echo "1,32,32,3"  | x2paddle -f onnx -m ../dataset/cifarnet/cifarnet.onnx -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
