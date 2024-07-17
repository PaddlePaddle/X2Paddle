# 进行转换
x2paddle -f tensorflow -m ../dataset/MobileNetV1/mobilenet_v1_1.0_224_frozen.pb -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
