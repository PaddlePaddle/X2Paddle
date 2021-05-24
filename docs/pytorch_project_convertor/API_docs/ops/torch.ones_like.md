## torch.ones_like
### [torch.ones_like](https://pytorch.org/docs/stable/generated/torch.ones_like.html?highlight=ones_like#torch.ones_like)

```python
torch.ones_like(input,
                *,
                dtype=None,
                layout=None,
                device=None,
                requires_grad=False,
                memory_format=torch.preserve_format)
```

### [paddle.ones_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ones_like_cn.html#ones-like)

```python
paddle.ones_like(x, dtype=None, name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x        | 表示输入Tensor。                                     |
| layout        | -            | 表示布局方式，PaddlePaddle无此参数。                   |
| device        | -            | 表示Tensor存放位置，PaddlePaddle无此参数。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle无此参数。 |
| memory_format   | -            | 表示内存格式，PaddlePaddle无此参数。           |  
