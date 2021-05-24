## torch.range

### [torch.range](https://pytorch.org/docs/stable/generated/torch.arange.html?highlight=arange#torch.range)
```python
torch.range(start=0,
            end,
            step=1,
            *,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False)
```
### [paddle.arange](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/arange_cn.html#arange)
```python
paddle.arange(start=0,
              end=None,
              step=1,
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
