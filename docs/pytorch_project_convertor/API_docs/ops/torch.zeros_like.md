## torch.zeros_like
### [torch.zeros_like](https://pytorch.org/docs/stable/generated/torch.zeros_like.html?highlight=zeros_like#torch.zeros_like)

```python
torch.zeros_like(input,
                 *,
                 dtype=None,
                 layout=None,
                 device=None,
                 requires_grad=False,
                 memory_format=torch.preserve_format)
```

### [paddle.zeros_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/zeros_like_cn.html#zeros-like)

```python
paddle.zeros_like(x, dtype=None, name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x        | 表示输入Tensor。                                     |
| layout        | -            | 表示布局方式，PaddlePaddle无此参数。                   |
| device        | -            | 表示Tensor存放位置，PaddlePaddle无此参数。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle无此参数。 |
| memory_format   | -            | 表示内存格式，PaddlePaddle无此参数。           |  
