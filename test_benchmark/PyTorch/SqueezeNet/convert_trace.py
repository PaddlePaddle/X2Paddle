import pickle
import torch
from torchvision.models import squeezenet1_0

with open('../dataset/SqueezeNet/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)["data0"]
torch_model = squeezenet1_0()
torch_state_dict = torch.load('../dataset/SqueezeNet/squeezenet1_0-b66bff10.pth')
torch_model.load_state_dict(torch_state_dict)
torch_model.eval()
save_dir = "pd_model_trace"
jit_type = "trace"

# print(torch_model(torch.tensor(input_data)))
import numpy as np
from x2paddle.convert import pytorch2paddle
pytorch2paddle(torch_model, save_dir, jit_type, [torch.tensor(input_data)], disable_feedback=True)
