# 进行转换
x2paddle -f caffe -p deploy_resnet18-priv.prototxt -w ../dataset/ResNet18/resnet18-priv.caffemodel -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
