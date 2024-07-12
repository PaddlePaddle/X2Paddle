# -*- coding: utf-8 -*-
# @Time : 20-6-3 下午5:39
# @Author : zhuying
# @Company : Minivision
# @File : train.py
# @Software : PyCharm

import argparse
import os
from MiniFASNet import MiniFASNetV1,MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
import torch
import numpy as np
from torchvision.models import AlexNet
from torchvision.models.utils import load_state_dict_from_url
from x2paddle.convert import pytorch2paddle
# 构建输入,pytorch-to-paddle
input_data = np.random.rand(1, 3, 80,80).astype("float32")
# np.save("input.npy", input_data)
#构建输入,pytorch-to-onnx
# input_data = torch.randn(1, 3, 80, 80)
# 获取PyTorch Module
torch_module = MiniFASNetV1SE(embedding_size=128, conv6_kernel=(5, 5),drop_p=0.75, num_classes=3, img_channel=3)
torch_state_dict = torch.load('../dataset/MiniFasNet/4_0_0_80x80_MiniFASNetV1SE.pth', map_location=torch.device('cpu'))
# torch_module = AlexNet()
# torch_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
torch_module.load_state_dict(torch_state_dict,False)
# 设置为eval模式
torch_module.eval()
# out = torch_module(torch.tensor(input_data))
# np.save("result.npy", out.detach().numpy())
# 进行转换,pytorch-to-paddle
pytorch2paddle(torch_module,
               save_dir="pd_model_trace",
               jit_type="trace",
               input_examples=[torch.tensor(input_data)], disable_feedback=True)
