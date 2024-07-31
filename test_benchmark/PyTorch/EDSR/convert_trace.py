import pickle
import torch
from edsr import EDSR
import numpy as np

torch_model = EDSR()
torch_model.load_state_dict(
    torch.load("../dataset/EDSR/model_best.pt",
               map_location=torch.device('cpu')))
torch_model.eval()

save_dir = "pd_model_trace"
jit_type = "trace"

img = np.load("../dataset/EDSR/input.npy")
from x2paddle.convert import pytorch2paddle

pytorch2paddle(torch_model,
               save_dir,
               jit_type, [torch.tensor(img)],
               disable_feedback=True)
