# 进行转换
x2paddle -f caffe -p det3.prototxt -w ../dataset/MTCNN3/det3.caffemodel -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
