# 进行转换
x2paddle --framework=tensorflow --model=../dataset/pix2pix/frozen_model.pb --save_dir=pd_model
# 运行推理程序
python pd_infer.py
