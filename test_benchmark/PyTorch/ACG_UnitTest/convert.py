import numpy as np

import torch
import torch.nn as nn

from x2paddle.convert import pytorch2paddle

from net import BasicTemporalModel


def main():
    model = BasicTemporalModel(in_channels=32,
                               num_features=512,
                               num_blocks=2,
                               time_window=3)

    ckpt = torch.load("../dataset/ACG_UnitTest/model_best.pth")
    state_dict = ckpt['model']
    model.load_state_dict(state_dict)

    model.eval()

    data = np.load("../dataset/ACG_UnitTest/input.npy")
    data = torch.Tensor(data)
    result = model(data)

    pytorch2paddle(module=model,
                   save_dir="./pd_model",
                   jit_type="trace",
                   input_examples=[data],
                   enable_code_optim=False)


if __name__ == '__main__':
    main()
