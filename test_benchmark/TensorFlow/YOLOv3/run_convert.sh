# 进行转换
echo "1,416,416,3" | x2paddle -f tensorflow -m ../dataset/YOLOv3/yolov3_coco.pb -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
