import pickle
import torch
from torchvision.models.segmentation import fcn_resnet50

with open('../dataset/FCN_ResNet50/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)["data0"]
torch_model = fcn_resnet50(pretrained=False, aux_loss=True, pretrained_backbone=False)
torch_state_dict = torch.load('../dataset/FCN_ResNet50/fcn_resnet50_coco-1167a1af.pth')
torch_model.load_state_dict(torch_state_dict)
torch_model.eval()
save_dir = "pd_model_script"
jit_type = "script"

import numpy as np
from x2paddle.convert import pytorch2paddle
pytorch2paddle(torch_model, save_dir, jit_type, [torch.tensor(input_data)], enable_code_optim=False, disable_feedback=True)
