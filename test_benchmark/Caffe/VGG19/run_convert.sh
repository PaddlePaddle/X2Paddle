# 进行转换
x2paddle -f caffe -p VGG_ILSVRC_19_layers_deploy.prototxt -w ../dataset/VGG19/VGG_ILSVRC_19_layers.caffemodel -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
