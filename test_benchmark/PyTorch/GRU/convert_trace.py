import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GruModel(nn.Module):
    """
    tsn
    """

    def __init__(self):
        super(GruModel, self).__init__()
        self.query_model = nn.GRU(768,
                                  256,
                                  batch_first=True,
                                  bidirectional=True)
        self.query_fc = nn.Linear(512, 256)

    def forward(self, query_input):
        """
        :param input:
        :param title_input:
        :return:
        """
        o1, h_query = self.query_model(query_input)
        new_query = []
        h_query_concat = torch.cat([h_query[0], h_query[1]], 1)
        new_query.append(h_query_concat[0])
        output_query = self.query_fc(torch.stack(new_query))
        output_query = F.normalize(output_query, p=2, dim=1)
        return output_query


grumodel = GruModel()
input_data = torch.rand(1, 5, 768)

# 设置为eval模式
grumodel.eval()
# 进行转换
from x2paddle.convert import pytorch2paddle

pytorch2paddle(grumodel,
               save_dir="pd_model_trace",
               jit_type="trace",
               input_examples=[input_data],
               disable_feedback=True)
