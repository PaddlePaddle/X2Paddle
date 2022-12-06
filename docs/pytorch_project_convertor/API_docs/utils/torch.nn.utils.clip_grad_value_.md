## torch.nn.utils.clip_grad_value_
### [torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html?highlight=clip_grad_value_#torch.nn.utils.clip_grad_value_)

```python
torch.nn.utils.clip_grad_value_(parameters, clip_value)
```

### [paddle.nn.ClipGradByValue](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/ClipGradByValue_cn.html#clipgradbyvalue)

```python
paddle.nn.ClipGradByValue(max, min=None)
```

### 参数差异

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| parameters   | -        | 表示要操作的 Tensor，PaddlePaddle 无此参数。  |
| clip_value   | -        | 表示裁剪梯度的范围，范围为 $[-clip_value, clip_vale]$，PaddlePaddle无此参数。  |
| - | min | 表示裁剪梯度的最小值，PyTorch 无此参数。  |
| - | max | 表示裁剪梯度的最小值，PyTorch 无此参数。  |

### 功能差异

#### 使用差异

***PyTorch***：属于原位操作，并且只裁剪固定范围 $[-clip_value, clip_vale]$
***PaddlePaddle***：需要实例化之后才可以使用，可自定义裁剪梯度的范围。

### 组合实现

用于梯度裁剪，PaddlePaddle 目前有对应 API，也可使用如下代码组合实现该 API。

```python
def clip_grad_value_(parameters, clip_value):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        paddle.clip(p.grad, min=-clip_value, max=clip_value)
```
