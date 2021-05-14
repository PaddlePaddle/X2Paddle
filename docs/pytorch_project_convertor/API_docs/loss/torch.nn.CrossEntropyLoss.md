## torch.nn.CrossEntropyLoss
### [torch.nn.CrossEntropyLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/loss/CrossEntropyLoss_cn.html#crossentropyloss)
```python
torch.nn.CrossEntropyLoss(weight=None,
                          size_average=None,
                          ignore_index=-100,
                          reduce=None,
                          reduction='mean')
```
### [paddle.nn.CrossEntropyLoss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/loss/CrossEntropyLoss_cn.html#crossentropyloss)
```python
paddle.nn.CrossEntropyLoss(weight=None,
                           ignore_index=-100,
                           reduction='mean',
                           soft_label=False,
                           axis=-1,
                           use_softmax=True,
                           name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size_average  | -        | PyTorch废弃参数。  |
| reduce  | -        | PyTorch废弃参数。  |
| -  | use_softmax        | 表示在使用交叉熵之前是否计算softmax，PyTorch无此参数。  |
| -  | soft_label        | 指明label是否为软标签，PyTorch无此参数。  |
| -  | axis        | 表示进行softmax计算的维度索引，PyTorch无此参数。  |

### 功能差异
#### 计算方式
***PyTorch***：只支持在使用交叉熵之前计算softmax且为硬标签的计算方式。  
***PaddlePaddle***：支持使用交叉熵之前是否计算softmax的设置，且支持软、硬标签两种计算方式，其计算方式可参见[文档](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/nn/layer/loss/CrossEntropyLoss_en.html)。
