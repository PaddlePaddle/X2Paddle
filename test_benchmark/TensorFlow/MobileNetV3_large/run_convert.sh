# 进行转换
x2paddle -f tensorflow -m ../dataset/MobileNetV3_large/v3-large_224_1.0_float.pb -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
