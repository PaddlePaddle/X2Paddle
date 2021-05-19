## torch.empty_like
### [torch.empty_like](https://pytorch.org/docs/stable/generated/torch.empty_like.html?highlight=empty_like#torch.empty_like)

```python
torch.empty_like(input,
                 *,
                 dtype=None,
                 layout=None,
                 device=None,
                 requires_grad=False,
                 memory_format=torch.preserve_format)
```

### [paddle.empty_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensor/creation/empty_like_cn.html#empty-like)

```python
paddle.empty_like(x, dtype=None, name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x        | 表示输入Tensor。                                     |
| layout        | -            | 表示布局方式，PaddlePaddle无此参数。                   |
| device        | -            | 表示Tensor存放位置，PaddlePaddle无此参数。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle无此参数。 |
| pin_memeory   | -            | 表示是否使用锁页内存，PaddlePaddle无此参数。           |  
