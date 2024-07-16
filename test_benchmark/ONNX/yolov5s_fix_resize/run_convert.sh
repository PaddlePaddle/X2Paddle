echo "N" | x2paddle -f onnx -m ../dataset/yolov5s_fix_resize/yolov5s-v12_sim.onnx -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
