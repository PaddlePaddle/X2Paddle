## torch.full_like
### [torch.full_like](https://pytorch.org/docs/stable/generated/torch.full_like.html?highlight=full_like#torch.full_like)

```python
torch.full_like(input,
                fill_value,
                *,
                dtype=None,
                layout=torch.strided,
                device=None,
                requires_grad=False,
                memory_format=torch.preserve_format)
```

### [paddle.full_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/full_like_cn.html#full-like)

```python
paddle.full_like(x, fill_value, dtype=None, name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x        | 表示输入Tensor。                                     |
| layout        | -            | 表示布局方式，PaddlePaddle无此参数。                   |
| device        | -            | 表示Tensor存放位置，PaddlePaddle无此参数。                   |
| requires_grad | -            | 表示是否阻断梯度传导，PaddlePaddle无此参数。 |
| memory_format   | -            | 表示是内存格式，PaddlePaddle无此参数。           |  
