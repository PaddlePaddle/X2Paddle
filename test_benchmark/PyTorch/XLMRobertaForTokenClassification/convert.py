import pickle
import torch
from transformers import RobertaForTokenClassification

with open('../dataset/XLMRobertaForTokenClassification/pytorch_input.pkl', 'rb') as inp:
    input_data = pickle.load(inp)
    
torch_model = RobertaForTokenClassification.from_pretrained("../dataset/XLMRobertaForTokenClassification/checkpoints", return_dict=False)

torch_model.eval()
save_dir = "pd_model"
jit_type = "trace"

from x2paddle.convert import pytorch2paddle
pytorch2paddle(torch_model, save_dir, jit_type, [torch.tensor(input_data["input_ids"]),
                                                 torch.tensor(input_data["attention_mask"])], disable_feedback=True)
