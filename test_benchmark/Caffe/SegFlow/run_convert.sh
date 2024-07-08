# 进行转换
x2paddle --framework=caffe --prototxt=deploy.prototxt --weight=../dataset/SegFlow/SegFlow.caffemodel --save_dir=pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
