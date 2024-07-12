# 进行转换
x2paddle -f caffe -p TSM.prototxt -w ../dataset/TSM/TSM.caffemodel -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
