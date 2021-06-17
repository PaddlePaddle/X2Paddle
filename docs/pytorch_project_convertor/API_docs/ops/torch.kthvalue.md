## torch.kthvalue

### [torch.kthvalue](https://pytorch.org/docs/stable/generated/torch.kthvalue.html?highlight=kthvalue#torch.kthvalue)

```python
torch.kthvalue(input, k, dim=None, keepdim=False, out=None) 
```
### 功能介绍
用于获取某一维度上第k大的数值，PaddlePaddle目前无对应API，可使用如下代码组合实现该API。
```python
import paddle

def kthvalue(input, k, dim=None, keepdim=False, out=None):
    values = paddle.sort(input, axis=dim)
    indices = paddle.argsort(input, axis=dim)
    values = paddle.slice(values, axes=[dim], starts=[k-1], ends=[k])
    indices = paddle.slice(indices, axes=[dim], starts=[k-1], ends=[k])
    if not keepdim:
        values = paddle.flatten(values)
        indices = paddle.flatten(indices)
    return values, indices
```

