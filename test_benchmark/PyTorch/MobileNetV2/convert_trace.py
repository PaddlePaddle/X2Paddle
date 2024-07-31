import pickle
import torch
from torchvision.models import MobileNetV2

with open('../dataset/MobileNetV2/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)["data0"]
torch_model = MobileNetV2()
torch_state_dict = torch.load(
    '../dataset/MobileNetV2/mobilenet_v2-b0353104.pth')
torch_model.load_state_dict(torch_state_dict)
torch_model.eval()
save_dir = "pd_model_trace"
jit_type = "trace"

import numpy as np
from x2paddle.convert import pytorch2paddle

pytorch2paddle(torch_model,
               save_dir,
               jit_type, [torch.tensor(input_data)],
               disable_feedback=True)
