from x2paddle.convert import pytorch2paddle

import torch
from models import __models__
import numpy as np

model = __models__['MSNet2D'](192)
# state_dict = torch.load('./MSNet2D_SF_DS_KITTI2015.ckpt', map_location=torch.device('cpu'))
state_dict = torch.load('../dataset/Mobilestereonet/MSNet2D_SF_DS_KITTI2015.ckpt')
param_dict = state_dict['model']
new_param_dict = {}
for k,v in param_dict.items():
    name = k[7:]
    new_param_dict[name] = v
model.load_state_dict(new_param_dict)

dummy_input_left = torch.randn(1, 3, 320, 480)
dummy_input_right = torch.randn(1, 3, 320, 480)

model.eval()

pytorch2paddle(module=model,
               save_dir="./pd_model",
               jit_type="trace",
               input_examples=[dummy_input_left, dummy_input_right],
               enable_code_optim=False)
