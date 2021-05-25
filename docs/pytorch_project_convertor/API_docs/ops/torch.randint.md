## torch.randint
### [torch.randint](https://pytorch.org/docs/stable/generated/torch.randint.html?highlight=randint#torch.randint)
```python
torch.randint(low=0,
              high,
              size,
              *,
              generator=None,
              out=None,
              dtype=None,
              layout=torch.strided,
              device=None,
              requires_grad=False)
```

### [paddle.randint](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/randint_cn.html#randint)
```python
paddle.randint(low=0,
               high=None,
               shape=[1],
               dtype=None,
               name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size          | shape        | 表示输出形状大小。                                     |
| generator        | -            | 用于采样的伪随机数生成器，PaddlePaddle无此参数。                   |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle无此参数。                   |
| device        | -            | 表示Tensor存放位置，PaddlePaddle无此参数。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle无此参数。 |


***【注意】*** 这类生成器的用法如下：
```python
G = torch.Generator()
G.manual_seed(1)
# 生成指定分布Tensor
torch.randperm(5, generator=G)
```
