## torch.optim.lr_scheduler.LRScheduler

### [torch.optim.lr_scheduler.LRScheduler](https://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html#LRScheduler)

```python
torch.optim.lr_scheduler.LRScheduler(optimizer, last_epoch=-1, verbose=False)
```

### [paddle.optimizer.lr.LRScheduler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LRScheduler_cn.html)

```python
paddle.optimizer.lr.LRScheduler(learning_rate=0.1, last_epoch=- 1, verbose=False)
```

### 参数差异

| PyTorch       | PaddlePaddle  | 备注                                                   |
| ------------- | ------------- | ----------------------------------------------------- |
| optimizer     | -             | 优化器, PaddlePaddle 无此参数。                          |
| -             | learning_rate | 学习率, Pytorch 无此参数。                               |

### 使用差异

#### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

model = nn.Linear(10, 10)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

#### PaddlePaddle

```python
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.optimizer.lr import StepDecay

model = nn.Linear(10, 10)
scheduler = StepDecay(learning_rate=0.1, step_size=30, gamma=0.1)
optimizer = optim.SGD(learning_rate=scheduler, parameters=model.parameters())

for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```
