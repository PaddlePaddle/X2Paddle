import pickle
import torch
from transformers import AutoModelForQuestionAnswering

with open('../dataset/CamembertForQuestionAnswering/pytorch_input.pkl',
          'rb') as inp:
    input_data = pickle.load(inp)

try:
    # the latest version of `transformers` support `scaled_dot_product_attention` (SDPA),
    # we can set `attn_implementation="eager"` not using it.
    torch_model = AutoModelForQuestionAnswering.from_pretrained(
        '../dataset/CamembertForQuestionAnswering/checkpoints',
        return_dict=False,
        attn_implementation="eager")
except:
    torch_model = AutoModelForQuestionAnswering.from_pretrained(
        '../dataset/CamembertForQuestionAnswering/checkpoints',
        return_dict=False)

torch_model.eval()
save_dir = "pd_model"
jit_type = "trace"

# out = torch_model(torch.tensor(input_data["input_ids"]), torch.tensor(input_data["attention_mask"]))

from x2paddle.convert import pytorch2paddle

pytorch2paddle(torch_model,
               save_dir,
               jit_type, [
                   torch.tensor(input_data["input_ids"]),
                   torch.tensor(input_data["attention_mask"])
               ],
               disable_feedback=True)
