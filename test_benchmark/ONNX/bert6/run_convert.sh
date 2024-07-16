# 进行转换
#echo "1,128\n1,128" | x2paddle -f onnx -m bert6.onnx -s pd_model_dygraph -df True
x2paddle -f onnx -m ../dataset/bert6/bert6.onnx -s pd_model_dygraph -df True -isd "{'input_x_word_0':[1,128], 'input_mask_0':[1,128]}"
# 运行推理程序
python pd_infer.py
