import pickle
import torch
from torchvision.models import MNASNet

with open('../dataset/MNasNet/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)["data0"]
torch_model = MNASNet(1.0)
torch_state_dict = torch.load('../dataset/MNasNet/mnasnet1.0_top1_73.512-f206786ef8.pth')
torch_model.load_state_dict(torch_state_dict)
torch_model.eval()
save_dir = "pd_model_script"
jit_type = "script"

import numpy as np
from x2paddle.convert import pytorch2paddle
pytorch2paddle(torch_model, save_dir, jit_type, [torch.tensor(input_data)], enable_code_optim=False, disable_feedback=True)
