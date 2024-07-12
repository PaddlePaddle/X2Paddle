import torch
import numpy as np
from x2paddle.convert import pytorch2paddle
from model_v3 import mobilenet_v3_large,mobilenet_v3_small


#配置
n_classes = 3
modelFile = '../dataset/mobilenetv3/MobileNetV3_large.pth'
torch_model = mobilenet_v3_large(num_classes=n_classes) #模型结构
#加载模型
checkpoint = torch.load(modelFile, map_location='cuda:0')
torch_model.load_state_dict(checkpoint)#加载预训练参数
# 设置为eval模式
torch_model.eval()

# 构建输入
input_data = np.load("../dataset/mobilenetv3/input.npy")

result = torch_model(torch.tensor(input_data))

# 进行转换
from x2paddle.convert import pytorch2paddle
pytorch2paddle(torch_model,
               save_dir="pd_model",
               jit_type="trace",
               input_examples=[torch.tensor(input_data)],
               disable_feedback=True,
            #    convert_to_lite=True,
            #    lite_valid_places="arm",
            #    lite_model_type="naive_buffer"
               )
               
# module (torch.nn.Module): PyTorch的Module
# save_dir (str): 转换后模型保存路径
# jit_type (str): 转换方式。目前有两种:trace和script,默认为trace
# input_examples (list[torch.tensor]): torch.nn.Module的输入示例，list的长度必须与输入的长度一致。默认为None。
# convert_to_lite (bool): 是否使用opt工具转成Paddle-Lite支持格式，默认为False
# lite_valid_places (str): 指定转换类型，可以同时指定多个backend(以逗号分隔)，opt将会自动选择最佳方式，默认为arm
# lite_model_type (str): 指定模型转化类型，目前支持两种类型：protobuf和naive_buffer，默认为naive_buffer
