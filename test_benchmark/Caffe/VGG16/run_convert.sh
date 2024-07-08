# 进行转换
x2paddle -f caffe -p vgg16.prototxt -w ../dataset/VGG16/VGG_ILSVRC_16_layers.caffemodel -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
