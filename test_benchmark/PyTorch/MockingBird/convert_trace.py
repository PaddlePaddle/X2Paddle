#import ipdb
import numpy as np
import os
from pathlib import Path
import torch
#import soundfile as sf
from x2paddle.convert import pytorch2paddle

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder.hifigan import inference as gan_vocoder
from vocoder.wavernn import inference as rnn_vocoder

import pickle

base_path = '.'
encoder_ckpt = Path(
    os.path.join(base_path,
                 '../dataset/MockingBird/encoder/saved_models/pretrained.pt'))
synthesizer_ckpt = Path(
    os.path.join(
        base_path,
        '../dataset/MockingBird/synthesizer/saved_models/mandarin/mandarin_200k.pt'
    ))
gan_vocoder_ckpt = Path(
    os.path.join(
        base_path,
        '../dataset/MockingBird/vocoder/saved_models/pretrained/g_hifigan.pt'))
rnn_vocoder_ckpt = Path(
    os.path.join(
        base_path,
        '../dataset/MockingBird/vocoder/saved_models/wavernn/wavernn_pretrained.pt'
    ))


# =========================== 1. Encoder ===========================
def convert_encoder():
    # 构建输入
    input_data = np.random.rand(4, 160, 40).astype(np.float32)
    # np.save("input_data_1019.npy", input_data)

    # 获取PyTorch Module
    encoder.load_model(encoder_ckpt)
    output = encoder._model.forward(torch.from_numpy(input_data).cuda())

    # with open("./result_SpeakerEncoder_pytorch.pkl", "wb") as fw:
    #     pickle.dump(output.cpu().detach().numpy(), fw)

    # 进行转换
    pytorch2paddle(
        encoder._model,
        save_dir="./pd_encoder_model_trace",
        jit_type="trace",
        # input_examples=[torch.tensor(input_data)])
        input_examples=[torch.tensor(input_data).cuda()],
        disable_feedback=True)


# # =========================== 2. Synthesizer ===========================
# def convert_synthesizer():
#     input_dict = dict()
#     # 构建输入
#     text = np.random.randint(1, 10, size=(1, 150), dtype=np.int64)
#     input_dict["text"] = text
#     text = torch.tensor(text)
#     mel = np.random.rand(1, 80, 2000).astype(np.float32)
#     input_dict["mel"] = mel
#     mel = torch.tensor(mel)
#     embed = np.random.rand(1, 256).astype(np.float32)
#     input_dict["embed"] = embed
#     embed = torch.tensor(embed)

#     # 获取PyTorch Module
#     synthesizer = Synthesizer(synthesizer_ckpt)
#     synthesizer.load()

#     # with open("./inputs_synthesizer_pytorch.pkl", "wb") as fw:
#     #     pickle.dump(input_dict, fw)
#     synthesizer._model.eval()
#     output = synthesizer._model.forward(text, mel, embed)
#     outputs_list = list()
#     for i in range(len(output)):
#         print(output[i].shape)
#         outputs_list.append(output[i].cpu().detach().numpy())
#     # with open("./result_synthesizer_pytorch.pkl", "wb") as fw:
#     #     pickle.dump(outputs_list, fw)
#     # torch.onnx.export(synthesizer._model, (text, mel, embed), "mynetwork.onnx", opset_version=12)

#     # 进行转换
#     pytorch2paddle(synthesizer._model,
#                 save_dir="./pd_synthesizer_model_trace_test_1107",
#                 jit_type="trace",
#                 input_examples=[text, mel, embed])
#                 # input_examples=[text.cuda(), mel.cuda(), embed.cuda()])


# =========================== 3. Vocoder ===========================
def convert_vocoder():
    # 构建输入
    input_data = np.random.rand(1, 80, 402).astype(np.float32)

    # np.save("input_data_1025.npy", input_data)
    # 获取PyTorch Module
    gan_vocoder.load_model(gan_vocoder_ckpt)

    gan_vocoder.generator.eval()
    output = gan_vocoder.generator.forward(torch.tensor(input_data).cuda())
    print("output.shape:", output.shape)

    # with open("./result_Vocoder_pytorch.pkl", "wb") as fw:
    #     pickle.dump(output.cpu().detach().numpy(), fw)

    # 进行转换
    pytorch2paddle(gan_vocoder.generator,
                   save_dir="./pd_vocoder_model_trace",
                   jit_type="trace",
                   input_examples=[torch.tensor(input_data).cuda()],
                   enable_code_optim=False,
                   disable_feedback=True)
    # input_examples=[torch.tensor(input_data).cuda()])


if __name__ == '__main__':
    # convert_encoder()
    # convert_synthesizer()
    convert_vocoder()
