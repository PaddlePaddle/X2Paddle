import pickle
import torch
from transformers import GPT2Model, GPT2Tokenizer

with open('../dataset/gpt2/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)

torch_model = GPT2Model.from_pretrained('../dataset/gpt2/checkpoints/',
                                        return_dict=False,
                                        attn_implementation="eager")

torch_model.eval()
save_dir = "pd_model"
jit_type = "trace"

from x2paddle.convert import pytorch2paddle

pytorch2paddle(torch_model,
               save_dir,
               jit_type, [
                   torch.tensor(input_data["input_ids"]),
               ],
               disable_feedback=True)
