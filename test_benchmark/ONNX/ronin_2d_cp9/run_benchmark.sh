# 进行转换
x2paddle -f onnx -m ../dataset/ronin_2d_cp9/ronin_2d_cp9.onnx -s pd_model
# 运行推理程序
python pd_infer.py
python benchmark_infer.py --use_gpu True --enable_trt True
python benchmark_infer.py --use_gpu True --enable_trt False
#python benchmark_infer.py --use_gpu False --enable_mkldnn True
python benchmark_infer.py --use_gpu False --enable_mkldnn False
