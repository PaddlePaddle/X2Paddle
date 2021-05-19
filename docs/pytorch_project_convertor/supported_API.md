# PyTorch训练项目转换支持API列表
> 目前PyTorch训练项目转换支持6个优化器相关API，70+的NN类API，10+Utils类API，2个Autograd类API，40+的基础操作API以及30+Torchvision API，我们在如下列表中给出了目前的全部API。

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
| 47   | torch.nn.functional.smooth_l1_loss                   | 48     |    torch.nn.AdaptiveAvgPool1d      |
| 49   | torch.nn.AdaptiveAvgPool2d                | 50     |    torch.nn.AdaptiveAvgPool3d      |
| 51   | torch.nn.AvgPool1d                   | 52     |    torch.nn.AvgPool2d      |
| 53   | torch.nn.AvgPool3d                   | 54     |    torch.nn.ConstantPad2d      |
| 55   | torch.nn.Dropout2d                   | 56     |    torch.nn.GELU      |
| 57   | torch.nn.GroupNorm                  | 58     |    torch.nn.Identity      |
| 59   | torch.nn.LayerNorm                | 60     |    torch.nn.MaxUnpool2d      |
| 61   | torch.nn.ReflectionPad2d                   | 62     |    torch.nn.ReplicationPad2d      |
| 63   | torch.nn.PReLU                   | 64     |    torch.nn.SyncBatchNorm      |
| 65   | torch.nn.ZeroPad2d                   | 66     |    torch.nn.KLDivLoss      |
| 67   | torch.nn.L1Loss                  | 68     |    paddle.nn.functional.interpolate     |
| 69   | torch.nn.functional.mse_loss              |   70  |    torch.nn.init.constant_    |
| 71   | torch.nn.init.normal_                   | 72     |    torch.nn.init.ones_     |
| 73   | torch.nn.init.zeros_                  | 74     |    torch.nn.init.orthogonal_     |



## Utils类API

| 序号 | API                            | 序号 | API                         |
| ---- | ------------------------------ | ---- | --------------------------- |
| 1    | torch.utils.data               | 2    | torch.utils.data.DataLoader |
| 3    | torch.utils.data.random_split  | 4    | torch.utils.data.Dataset    |
| 5    | torch.utils.data.ConcatDataset | 6     | torch.utils.data.distributed      |
| 7    | torch.utils.data.distributed.DistributedSampler  | 8    | torch.utils.model_zoo |
| 9    | torch.utils.model_zoo.load_url  | 10    | torch.multiprocessing   |
| 11    | torch.multiprocessing.spawn              | 12    | torch.distributed |
| 13    | torch.distributed.init_process_group  | 14    |  |

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
| 43   | torch.randn             | 44   | torch.add |
| 45   | torch.mul          | 46   | torch.linspace  |
| 47   | torch.einsum|     |        |

## Torchvision API

| 序号 | API                               | 序号 | API                                         |
| ---- | --------------------------------- | ---- | ------------------------------------------- |
| 1    | torchvision.transforms            | 2    | torchvision.transforms.Compose              |
| 3    | torchvision.transforms.ToPILImage | 4    | torchvision.transforms.Resize               |
| 5    | torchvision.transforms.ToTensor   | 6    | torchvision.transforms.RandomHorizontalFlip |
| 7    | torchvision.transforms.CenterCrop | 8    | torchvision.transforms.Normalize            |
| 9    | torchvision.utils.save_image      | 10   | torchvision.datasets.ImageFolder            |
| 11    | torchvision.transforms.RandomResizedCrop   | 12    | torchvision.transforms.Lambda              |
| 13    | torchvision.utils | 14    | torchvision.utils.save_image               |
| 15    | torchvision.datasets   | 16    | torchvision.datasets.ImageFolder |
| 17    | torchvision.models | 18    | torchvision.models.vgg_pth_urls           |
| 19    | torchvision.models.vgg11      | 20   | torchvision.models.vgg13            |
| 21    | torchvision.models.vgg16   | 22    | torchvision.models.vgg19             |
| 23    | torchvision.models.vgg11_bn | 24    | torchvision.models.vgg13_bn        |
| 25    | torchvision.models.vgg16_bn   | 26    | torchvision.models.vgg19_bn |
| 27    | torchvision.models.resnet34 | 28    | torchvision.models.resnet50       |
| 29    | torchvision.models.resnet101     | 30   | torchvision.models.resnet152            |
| 31    | torchvision.models.resnext50_32x4d   | 32    | torchvision.models.resnext101_32x8d       |
| 33    | torchvision.models.wide_resnet50_2 | 34    | torchvision.models.wide_resnet101_2      |


***持续更新...***
