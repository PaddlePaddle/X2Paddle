## torch.nn.utils.clip_grad_value_
### [torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html?highlight=clip_grad_value_#torch.nn.utils.clip_grad_value_)

```python
torch.nn.utils.clip_grad_value_(parameters, clip_value)
```

### 功能介绍
用于梯度裁剪，PaddlePaddle目前无对应API，可使用如下代码组合实现该API。
```python
def clip_grad_value_(parameters, clip_value):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        paddle.clip(p.grad, min=-clip_value, max=clip_value)
```
