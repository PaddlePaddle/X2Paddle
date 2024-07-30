## torch.utils.data.BatchSampler
### [torch.utils.data.BatchSampler](https://pytorch.org/docs/stable/data.html?highlight=batchsampler#torch.utils.data.BatchSampler)
```python
torch.utils.data.BatchSampler(sampler, batch_size, drop_last)
```

### [paddle.io.BatchSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/BatchSampler_cn.html#batchsampler)
```python
paddle.io.BatchSampler(dataset=None, sampler=None, shuffle=Fasle, batch_size=1, drop_last=False)
```

### 功能差异
#### 使用方式
***PyTorch***：只能使用`sampler`来构建BatchSampler。
***PaddlePaddle***：能使用`sampler`和`dataset`来构建BatchSampler。
