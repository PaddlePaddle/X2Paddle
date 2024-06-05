# 进行转换
x2paddle -f caffe -p shufflenet_v2_x0.5.prototxt -w shufflenet_v2_x0.5.caffemodel -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
