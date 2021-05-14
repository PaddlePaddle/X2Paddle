## torch.nn.KLDivLoss
### [torch.nn.KLDivLoss](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html?highlight=kldivloss#torch.nn.KLDivLoss)
```python
torch.nn.KLDivLoss(size_average=None,
                   reduce=None,
                   reduction='mean',
                   log_target=False)
```

### [paddle.nn.KLDivLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/loss/KLDivLoss_cn.html)
```python
paddle.nn.KLDivLoss(reduction='mean')
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -        | PyTorch废弃参数。  |
| reduce  | -        | PyTorch废弃参数。  |
| log_target  | -        | 表示是否对目标值进行log处理，PaddlePaddle无此参数。  |

### 功能差异
#### 计算方式
***PyTorch***：  
> 当`log_target`为`True`时：  
> $ l(input,label)= e^{target}∗(label−input) $  
>  
> 当`log_target`为`False`时：  
> 1. $ l(input,label)=target*(log(target)-input) $  
> 2. $ l(input,label) $中值小于0的取0  

***PaddlePaddle***：
> $ l(input,label)=label∗(log(label)−input) $

在PaddlePaddle中可使用如下代码组合实现该API。  
```python
import paddle

# 定义KLDivLoss
class KLDivLoss(paddle.nn.Layer):
    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction='mean',
                 log_target=False):
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input, target):
        if self.log_target:
            out = paddle.exp(target) * (target - input)
        else:
            out_pos = target * (paddle.log(target) - input)
            zeros = paddle.zeros_like(out_pos)
            out = paddle.where(target > 0, out_pos, zeros)
        out_sum = paddle.sum(out)
        if self.reduction == "sum":
            return out_sum
        elif self.reduction == "batchmean":
            n = input.shape[0]
            return out_sum / n
        elif self.reduction == "mean":
            return paddle.mean(out)
        else:
            return out

# 构造输入  
import numpy as np
shape = (5, 20)
x = np.random.uniform(-10, 10, shape).astype('float32')
target = np.random.uniform(-10, 10, shape).astype('float32')

# 计算loss
kldiv_criterion = KLDivLoss()
pred_loss = kldiv_criterion(paddle.to_tensor(x),
                            paddle.to_tensor(target))
```
