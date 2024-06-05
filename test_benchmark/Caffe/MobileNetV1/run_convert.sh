# 进行转换
x2paddle -f caffe -p mobilenet_deploy.prototxt -w mobilenet.caffemodel -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
