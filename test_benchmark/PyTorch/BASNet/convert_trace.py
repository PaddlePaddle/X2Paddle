import pickle
import torch
from model import BASNet


# import numpy as np
# ipt = np.random.rand(1, 3, 224, 224).astype("float32")
f = open('../dataset/BASNet/input.pkl', 'rb')
ipt = pickle.load(f)["input"]
f.close()
net = BASNet(3,1)
net.load_state_dict(torch.load("../dataset/BASNet/basnet.pth"))
net.eval()


save_dir = "pd_model_trace"
jit_type = "trace"

from x2paddle.convert import pytorch2paddle
pytorch2paddle(net, save_dir, jit_type, [torch.tensor(ipt)], disable_feedback=True)
