import pickle
import torch
import easyocr
import numpy as np

reader = easyocr.Reader(
    ['ch_sim', 'en'],
    model_storage_directory="../dataset/EasyOCR_recognizer/models",
    download_enabled=False)
torch_model = reader.recognizer
torch_model.eval()

save_dir = "pd_model_trace"
jit_type = "trace"
has_cuda = torch.cuda.is_available()

img = np.load("../dataset/EasyOCR_recognizer/img.npy")
text = np.load("../dataset/EasyOCR_recognizer/text.npy")
from x2paddle.convert import pytorch2paddle
if has_cuda:
    pytorch2paddle(torch_model.module,
                   save_dir,
                   jit_type,
                   [torch.tensor(img).cuda(),
                    torch.tensor(text).cuda()],
                   disable_feedback=True)
else:
    pytorch2paddle(torch_model,
                   save_dir,
                   jit_type,
                   [torch.tensor(img), torch.tensor(text)],
                   disable_feedback=True)
