# 进行转换
x2paddle -f caffe -p 3class.prototxt -w 3class.caffemodel -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
