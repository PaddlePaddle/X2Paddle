# 进行转换
x2paddle -f tensorflow -m ../dataset/InceptionResNetV2/inception_resnet_v2.pb -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
