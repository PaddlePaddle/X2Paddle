import pickle
import torch
from resnet18 import get_res_pvnet
import numpy as np

torch_model = get_res_pvnet(18, 2, 13)
torch_model.load_state_dict(
    torch.load("../dataset/ResNet18_2/189.pth",
               map_location=torch.device('cpu'))["net"])
torch_model.eval()
save_dir = "pd_model_trace"
jit_type = "trace"

np.random.seed(6)
input = np.random.rand(1, 5, 480, 640).astype("float32")

from x2paddle.convert import pytorch2paddle

pytorch2paddle(torch_model,
               save_dir,
               jit_type, [torch.tensor(input)],
               disable_feedback=True)
