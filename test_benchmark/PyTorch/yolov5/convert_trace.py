import torch
import numpy as np

torch_model = torch.hub.load('../dataset/yolov5/yolov5',
                             model='custom',
                             source='local',
                             path='../dataset/yolov5/yolov5s.pt')
torch_model.eval()

img = np.load("../dataset/yolov5/input.npy")

save_dir = "pd_model_trace"
jit_type = "trace"

# https://github.com/ultralytics/yolov5/issues/9341
# torch need to load yolo model twice!!!
# so, we load one time before converting to paddle!!!
try:
    torch.jit.trace(torch_model, torch.tensor(img))
except:
    pass

from x2paddle.convert import pytorch2paddle

pytorch2paddle(torch_model,
               save_dir,
               jit_type, [torch.tensor(img)],
               disable_feedback=True)
