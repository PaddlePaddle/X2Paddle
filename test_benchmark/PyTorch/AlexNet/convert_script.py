import pickle
import torch
from torchvision.models import AlexNet

with open('../dataset/AlexNet/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)["data0"]
torch_model = AlexNet()
torch_state_dict = torch.load('../dataset/AlexNet/alexnet-owt-4df8aa71.pth')
torch_model.load_state_dict(torch_state_dict)
torch_model.eval()

save_dir = "pd_model_script"
jit_type = "script"

from x2paddle.convert import pytorch2paddle

pytorch2paddle(torch_model,
               save_dir,
               jit_type, [torch.tensor(input_data)],
               disable_feedback=True)
