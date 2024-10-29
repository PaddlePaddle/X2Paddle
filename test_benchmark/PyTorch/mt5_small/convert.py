import pickle
import torch
from transformers import MT5Model, MT5Tokenizer

with open('../dataset/mt5_small/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)

torch_model = MT5Model.from_pretrained('../dataset/mt5_small/checkpoints/',
                                       return_dict=False,
                                       attn_implementation="eager")

torch_model.eval()
save_dir = "pd_model"
jit_type = "trace"

from x2paddle.convert import pytorch2paddle

pytorch2paddle(torch_model,
               save_dir,
               jit_type, {
                   "input_ids": torch.tensor(input_data[0]),
                   "decoder_input_ids": torch.tensor(input_data[1]),
               },
               disable_feedback=True)
