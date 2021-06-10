## torch.var_mean

### [torch.var_mean](https://pytorch.org/docs/stable/generated/torch.var_mean.html?highlight=var_mean#torch.var_mean)
```python
# 用法一：
torch.var_mean(input, unbiased=True)
# 用法二：
torch.var_mean(input, dim, keepdim=False, unbiased=True)
```

### 功能介绍
用于实现返回Tensor的方差和均值，PaddlePaddle目前无对应API，可使用如下代码组合实现该API。

```python
import paddle

def var_mean(input, dim=None, unbiased=True, keepdim=False):
    var = paddle.var(input, axis=dim, 
                     unbiased=unbiased, keepdim=keepdim)
    mean = paddle.mean(input, 
                       axis=dim, 
                       keepdim=keepdim)
    return var, mean
```
