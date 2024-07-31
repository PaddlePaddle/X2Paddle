# 进行转换
echo "None,224,224,3" | x2paddle -f tensorflow -m ../dataset/MobileNetV2/mobilenet_v2_1.4_224_frozen.pb -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
python benchmark_infer.py --use_gpu True --enable_trt True
python benchmark_infer.py --use_gpu True --enable_trt False
python benchmark_infer.py --use_gpu False --enable_mkldnn True
python benchmark_infer.py --use_gpu False --enable_mkldnn False
