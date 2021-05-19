## torch.utils.data.distributed.DistributedSampler
### [torch.utils.data.distributed.DistributedSampler](https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler)
```python
torch.utils.data.distributed.DistributedSampler(dataset,
                                                num_replicas=None,
                                                rank=None,
                                                shuffle=True,
                                                seed=0,
                                                drop_last=False)
```

### 功能介绍
用于实现分布式数据采集器，PaddlePaddle目前无对应API，可使用如下代码组合实现该API。
```python
import paddle
class DistributedSampler(paddle.io.DistributedBatchSampler):
    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0,
                 drop_last=False):
        super().__init__(
            dataset=dataset,
            batch_size=1,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last)
```
