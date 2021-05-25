## torch.linspace
### [torch.linspace](https://pytorch.org/docs/stable/generated/torch.linspace.html?highlight=linspace#torch.linspace)
```python
torch.linspace(start,
               end,
               steps,
               *,
               out=None,
               dtype=None,
               layout=torch.strided,
               device=None,
               requires_grad=False)
```

### [paddle.linspace](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linspace_cn.html#linspace)
```python
paddle.linspace(start,
                stop,
                num,
                dtype=None,
                name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle无此参数。                   |
| device        | -            | 表示Tensor存放位置，PaddlePaddle无此参数。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle无此参数。 |
