## Loss类API映射列表
该文档梳理了计算loss相关的PyTorch-PaddlePaddle API映射列表。
| 序号 | PyTorch API                                                  | PaddlePaddle API                                             | 备注                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [torch.nn.L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html?highlight=l1loss#torch.nn.L1Loss) | [paddle.nn.loss.L1Loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/loss/L1Loss_cn.html#l1loss) | 功能一致，PyTroch存在废弃参数`size_average`和`reduce`。      |
| 2    | [torch.nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html?highlight=mseloss#torch.nn.MSELoss) | [paddle.nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html?highlight=mseloss#torch.nn.MSELoss) | 功能一致，PyTroch存在废弃参数`size_average`和`reduce`。      |
| 3    | [torch.nn.CrossEntropyLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/loss/CrossEntropyLoss_cn.html#crossentropyloss) | [paddle.nn.CrossEntropyLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/loss/CrossEntropyLoss_cn.html#crossentropyloss) | [差异对比](torch.nn.CrossEntropyLoss.md)                |
| 4    | [torch.nn.KLDivLoss](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html?highlight=kldivloss#torch.nn.KLDivLoss) | [paddle.nn.KLDivLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/loss/KLDivLoss_cn.html) | [差异对比](torch.nn.KLDivLoss.md)                       |
| 5    | [torch.nn.BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html?highlight=bceloss#torch.nn.BCELoss) | [paddle.nn.BCELoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/loss/BCELoss_cn.html#bceloss) | 功能一致，PyTroch存在废弃参数`size_average`和`reduce`。      |
| 6    | [torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bcewithlogitsloss#torch.nn.BCEWithLogitsLoss) | [paddle.nn.BCEWithLogitsLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/loss/BCEWithLogitsLoss_cn.html#bcewithlogitsloss) | 功能一致，PyTroch存在废弃参数`size_average`和`reduce`。      |
| 7    | [torch.nn.SmoothL1Loss](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html?highlight=torch%20nn%20smoothl1loss#torch.nn.SmoothL1Loss) | [paddle.nn.SmoothL1Loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/loss/SmoothL1Loss_cn.html#smoothl1loss) | 功能一致，参数名不一致，PyTroch存在废弃参数`size_average`和`reduce`。 |


***持续更新...***
