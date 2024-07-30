import pickle
import torch
from transformers import AutoModel

with open('../dataset/BertModel_SpanBert/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)

torch_model = AutoModel.from_pretrained(
    "../dataset/BertModel_SpanBert/checkpoints", return_dict=False)

torch_model.eval()
save_dir = "pd_model"
jit_type = "trace"

from x2paddle.convert import pytorch2paddle

pytorch2paddle(torch_model,
               save_dir,
               jit_type, [
                   torch.tensor(input_data["input_ids"]),
                   torch.tensor(input_data["attention_mask"]),
                   torch.tensor(input_data["token_type_ids"])
               ],
               disable_feedback=True)
