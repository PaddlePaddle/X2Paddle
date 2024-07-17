# 进行转换
echo "None,512,512,1\n1" | x2paddle -f tensorflow -m ../dataset/ToyUNet-ToyUnet/toy_unet.pb -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py
