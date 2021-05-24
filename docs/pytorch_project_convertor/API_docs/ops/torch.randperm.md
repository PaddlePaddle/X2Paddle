## torch.randperm
### [torch.randperm](https://pytorch.org/docs/stable/generated/torch.randperm.html?highlight=randperm#torch.randperm)
```python
torch.randperm(n,
               *,
               generator=None,
               out=None,
               dtype=torch.int64,
               layout=torch.strided,
               device=None,
               requires_grad=False,
               pin_memory=False)
```
### [paddle.randperm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/randperm_cn.html#randperm)
```python
paddle.randperm(n, dtype='int64', name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| generator        | -            | 用于采样的伪随机数生成器，PaddlePaddle无此参数。                   |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle无此参数。                   |
| device        | -            | 表示Tensor存放位置，PaddlePaddle无此参数。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle无此参数。 |
| pin_memeory   | -            | 表示是否使用锁页内存，PaddlePaddle无此参数。           |  


***【注意】*** 这类生成器的用法如下：
```python
G = torch.Generator()
G.manual_seed(1)
# 生成指定分布Tensor
torch.randperm(5, generator=G)
```
