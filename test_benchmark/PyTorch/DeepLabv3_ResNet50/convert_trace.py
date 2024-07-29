import pickle
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

with open('../dataset/DeepLabv3_ResNet50/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)["data0"]
# torch_model = deeplabv3_resnet50(pretrained=True)
torch_model = deeplabv3_resnet50(pretrained_backbone = False,aux_loss = True)
torch_state_dict = torch.load('../dataset/DeepLabv3_ResNet50/deeplabv3_resnet50_coco-cd0a2569.pth')
torch_model.load_state_dict(torch_state_dict)
torch_model.eval()
save_dir = "pd_model_trace"
jit_type = "trace"

import numpy as np
from x2paddle.convert import pytorch2paddle
pytorch2paddle(torch_model, save_dir, jit_type, [torch.tensor(input_data)], disable_feedback=True)
