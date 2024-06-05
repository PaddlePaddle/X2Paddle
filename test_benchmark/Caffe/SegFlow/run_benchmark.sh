# 进行转换
x2paddle --framework=caffe --prototxt=deploy.prototxt --weight=SegFlow.caffemodel --save_dir=pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
python benchmark_infer.py --use_gpu True --enable_trt True
python benchmark_infer.py --use_gpu True --enable_trt False
python benchmark_infer.py --use_gpu False --enable_mkldnn True
python benchmark_infer.py --use_gpu False --enable_mkldnn False