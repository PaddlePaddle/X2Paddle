# PyTorch训练项目转换支持API列表
> 目前PyTorch训练项目转换支持6个优化器相关API，40+的NN类API，5个Utils类API，2个Autograd类API，40+的基础操作API以及10+Torchvision API，我们在如下列表中给出了目前的全部API。

## 优化器相关API
| 序号 | API                                        | 序号 | API                                        |
| ---- | ------------------------------------------ | ---- | ------------------------------------------ |
| 1    | torch.optim                                | 2    | torch.optim.lr_scheduler.ReduceLROnPlateau |
| 3    | torch.optim.lr_scheduler.CosineAnnealingLR | 4    | torch.optim.lr_scheduler.MultiStepLR       |
| 5    | torch.optim.Adam                           | 6    | torch.optim.SGD                            |

## NN类API

| 序号 | API                                                  | 序号 | API                               |
| ---- | ---------------------------------------------------- | ---- | --------------------------------- |
| 1    | torch.nn                                             | 2    | torch.nn.Module                   |
| 3    | torch.nn.ModuleList                                  | 4    | torch.nn.Sequential               |
| 5    | torch.nn.utils                                       | 6    | torch.nn.utils.clip_grad_value_   |
| 7    | torch.nn.Parameter                                   | 8    | torch.nn.DataParallel             |
| 9    | torch.nn.functional                                  | 10   | torch.nn.BatchNorm1d              |
| 11   | torch.nn.BatchNorm2d                                 | 12   | torch.nn.BatchNorm3d              |
| 13   | torch.nn.Conv1d                                      | 14   | torch.nn.Conv2d                   |
| 15   | torch.nn.Conv3d                                      | 16   | torch.nn.ConvTranspose2d          |
| 17   | torch.nn.Dropout                                     | 18   | torch.nn.Embedding                |
| 19   | torch.nn.InstanceNorm2d                              | 20   | torch.nn.LeakyReLU                |
| 21   | torch.nn.Linear                                      | 22   | torch.nn.MaxPool1d                |
| 23   | torch.nn.MaxPool2d                                   | 24   | torch.nn.MaxPool3d                |
| 25   | torch.nn.ReLU                                        | 26   | torch.nn.Sigmoid                  |
| 27   | torch.nn.Softmax                                     | 28   | torch.nn.Tanh                     |
| 29   | torch.nn.Upsample                                    | 30   | torch.nn.CrossEntropyLoss         |
| 31   | torch.nn.BCEWithLogitsLoss                           | 32   | torch.nn.BCELoss                  |
| 33   | torch.nn.functional.avg_pool1d                       | 34   | torch.nn.functional.avg_pool2d    |
| 35   | torch.nn.functional.avg_pool3d                       | 36   | torch.nn.functional.dropout       |
| 37   | torch.nn.functional.log_softmax                      | 38   | torch.nn.functional.pad           |
| 39   | torch.sigmoid                                        | 40   | torch.nn.functional.sigmoid       |
| 41   | torch.nn.functional.softmax                          | 42   | torch.nn.init.xavier_uniform_     |
| 43   | torch.nn.functional.binary_cross_entropy_with_logits | 44   | torch.nn.functional.cross_entropy |
| 45   | torch.nn.functional.dropout                          | 46   | torch.nn.functional.relu          |
| 47   | torch.nn.functional.smooth_l1_loss                   |      |                                   |

## Utils类API

| 序号 | API                            | 序号 | API                         |
| ---- | ------------------------------ | ---- | --------------------------- |
| 1    | torch.utils.data               | 2    | torch.utils.data.DataLoader |
| 3    | torch.utils.data.random_split  | 4    | torch.utils.data.Dataset    |
| 5    | torch.utils.data.ConcatDataset |      |                             |

## Autograd类API

| 序号 | API                     | 序号 | API                 |
| ---- | ----------------------- | ---- | ------------------- |
| 1    | torch.autograd.Variable | 2    | torch.autograd.grad |

## 基础操作API

| 序号 | API                     | 序号 | API                     |
| ---- | ----------------------- | ---- | ----------------------- |
| 1    | torch                   | 2    | torch.Tensor            |
| 3    | torch.FloatTensor       | 4    | torch.load              |
| 5    | torch.save              | 6    | torch.device            |
| 7    | torch.cat               | 8    | torch.cuda.is_available |
| 9    | torch.no_grad           | 10   | torch.from_numpy        |
| 11   | torch.cuda.device_count | 12   | torch.manual_seed       |
| 13   | torch.unsqueeze         | 14   | torch.squeeze           |
| 15   | torch.sum               | 16   | torch.mean              |
| 17   | torch.full              | 18   | torch.full_like         |
| 19   | torch.ones              | 20   | torch.ones_like         |
| 21   | torch.zeros             | 22   | torch.zeros_like        |
| 23   | torch.sqrt              | 24   | torch.arange            |
| 25   | torch.matmul            | 26   | torch.set_grad_enabled  |
| 27   | torch.tensor            | 28   | torch.clamp             |
| 29   | torch.exp               | 30   | torch.max               |
| 31   | torch.min               | 32   | torch.argmax            |
| 33   | torch.argmin            | 34   | torch.stack             |
| 35   | torch.log               | 36   | torch.randperm          |
| 37   | torch.rand              | 38   | torch.abs               |
| 39   | torch.bitwise_or        | 40   | torch.bitwise_xor       |
| 41   | torch.bitwise_and       | 42   | torch.bitwise_not       |

## Torchvision API

| 序号 | API                               | 序号 | API                                         |
| ---- | --------------------------------- | ---- | ------------------------------------------- |
| 1    | torchvision.transforms            | 2    | torchvision.transforms.Compose              |
| 3    | torchvision.transforms.ToPILImage | 4    | torchvision.transforms.Resize               |
| 5    | torchvision.transforms.ToTensor   | 6    | torchvision.transforms.RandomHorizontalFlip |
| 7    | torchvision.transforms.CenterCrop | 8    | torchvision.transforms.Normalize            |
| 9    | torchvision.utils.save_image      | 10   | torchvision.datasets.ImageFolder            |

***持续更新...***