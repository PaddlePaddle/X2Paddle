## torch.full

### [torch.full](https://pytorch.org/docs/stable/generated/torch.full.html?highlight=full#torch.full)
```python
torch.full(size,
           fill_value,
           *,
           out=None,
           dtype=None,
           layout=torch.strided,
           device=None,
           requires_grad=False)
```

### [paddle.full](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/full_cn.html#full)
```python
paddle.full(shape,
            fill_value,
            dtype=None,
            name=None)
```


### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size          | shape        | 表示输出形状大小。                                     |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle无此参数。                   |
| device        | -            | 表示Tensor存放位置，PaddlePaddle无此参数。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle无此参数。 |
