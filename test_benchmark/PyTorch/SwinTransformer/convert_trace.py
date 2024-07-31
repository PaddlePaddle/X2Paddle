import torch
import numpy as np
from models.swin_transformer import SwinTransformer
# 构建输入
input_data = np.random.rand(1, 3, 224, 224).astype("float32")

swin_model_cfg_map = {
    "swin_tiny_patch4_window7_224": {
        "EMBED_DIM": 96,
        "DEPTHS": [2, 2, 6, 2],
        "NUM_HEADS": [3, 6, 12, 24],
        "WINDOW_SIZE": 7,
    }
}

model_name = "swin_tiny_patch4_window7_224"
torch_module = SwinTransformer(**swin_model_cfg_map[model_name])
torch_state_dict = torch.load(
    "../dataset/SwinTransformer/{}.pth".format(model_name))["model"]
torch_module.load_state_dict(torch_state_dict)
model_name = "pd_model_trace"
# 设置为eval模式
torch_module.eval()
# 进行转换
from x2paddle.convert import pytorch2paddle

pytorch2paddle(torch_module,
               save_dir=model_name,
               jit_type="trace",
               input_examples=[torch.tensor(input_data)],
               disable_feedback=True)

# np.save("input.npy", input_data)
# np.save("output.npy", torch_module(torch.tensor(input_data)).detach().numpy())
