import torch
import numpy as np
import os
from model import Net

input_data = np.load("../dataset/opadd/ipt.npy")
torch_model = Net(16, 4)
torch_state_dict = torch.load("../dataset/opadd/rand.pth")
torch_model.load_state_dict(torch_state_dict, strict=False)
torch_model.eval()
save_dir = "pd_model_trace"
jit_type = "trace"

from x2paddle.convert import pytorch2paddle

pytorch2paddle(torch_model,
               save_dir,
               jit_type, [torch.tensor(input_data)],
               disable_feedback=True)
