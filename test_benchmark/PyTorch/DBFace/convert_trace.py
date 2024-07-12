import common
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from model.DBFace import DBFace

HAS_CUDA = torch.cuda.is_available()

dbface = DBFace()
dbface.eval()
if HAS_CUDA:
    dbface.cuda()
dbface.load("../dataset/DBFace/model/dbface.pth")    
file = "../dataset/DBFace/datas/selfie.jpg"
image = common.imread(file)
mean = [0.408, 0.447, 0.47]
std = [0.289, 0.274, 0.278]
image = common.pad(image)
image = ((image / 255.0 - mean) / std).astype(np.float32)
image = image.transpose(2, 0, 1)
torch_image = torch.from_numpy(image)[None]
if HAS_CUDA:
    torch_image = torch_image.cuda()
# res = dbface(torch_image)
# res_dict = dict()
# import pickle
# res_dict["hm"] = res[0].detach().numpy()
# res_dict["box"] = res[1].detach().numpy()
# res_dict["landmark"] = res[2].detach().numpy()
# fileObject = open("outputs.pkl", 'wb')
# pickle.dump(res_dict, fileObject)
# fileObject.close()    

save_dir = "pd_model_trace"
jit_type = "trace"

from x2paddle.convert import pytorch2paddle
pytorch2paddle(dbface, save_dir, jit_type, [torch_image], disable_feedback=True)
