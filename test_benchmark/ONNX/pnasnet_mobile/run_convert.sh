# 进行转换
echo "-1,224,224,3"  | x2paddle -f onnx -m ../dataset/pnasnet_mobile/pnasnet_mobile.onnx -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
