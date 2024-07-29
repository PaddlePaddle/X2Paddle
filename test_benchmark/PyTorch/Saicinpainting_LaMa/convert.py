import torch
import numpy as np
import os
import sys

from saicinpainting.training.modules.ffc import FFCResNetGenerator

# build input
input_data = np.random.rand(1, 4, 1024, 1024).astype("float32")

# build pytorch model
model = FFCResNetGenerator(input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9,
                 init_conv_kwargs={'ratio_gin':0, 'ratio_gout':0, 'enable_lfu':False}, 
                 downsample_conv_kwargs={'ratio_gin':0, 'ratio_gout':0, 'enable_lfu':False}, 
                 resnet_conv_kwargs={'ratio_gin':0.75, 'ratio_gout':0.75, 'enable_lfu':False},
                 add_out_act='sigmoid')

state = torch.load('../dataset/Saicinpainting_LaMa/lama-fourier/models/best.ckpt')

model_dict = model.state_dict()
pretrain_dict = {}
for k,v in model_dict.items():
    pretrain_dict[k] = state['state_dict']['generator.'+k]
model_dict.update(pretrain_dict)
model.load_state_dict(model_dict)
#model.load_state_dict(pretrain_dict)
# from torchvision.models import AlexNet
# torch_model = AlexNet()
# torch_state_dict = torch.load('alexnet-owt-4df8aa71.pth')
# torch_model.load_state_dict(torch_state_dict)

# 设置为eval模式
model.eval()
out = model(torch.from_numpy(input_data))
print("done")

from x2paddle.convert import pytorch2paddle

pytorch2paddle(model,
               save_dir="pd_model_trace/",
               jit_type="trace",
               input_examples=[torch.tensor(input_data)], disable_feedback=True)
