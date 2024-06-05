# 进行转换
x2paddle -f caffe -p deploy_MnasNet.prototxt -w MnasNet_model_cat_dog_iter_64000.caffemodel -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
