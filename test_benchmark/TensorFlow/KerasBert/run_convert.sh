# 进行转换
echo "None,128\nNone,128" | x2paddle -f tensorflow -m ../dataset/KerasBert/model.pb -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
