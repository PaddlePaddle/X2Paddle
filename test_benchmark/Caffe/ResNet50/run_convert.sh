# 进行转换
x2paddle -f caffe -p deploy_1.prototxt -w ResNet-50-model.caffemodel -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
