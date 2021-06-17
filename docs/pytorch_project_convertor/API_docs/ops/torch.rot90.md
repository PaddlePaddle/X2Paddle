## torch.rot90

### [torch.rot90](https://pytorch.org/docs/stable/generated/torch.rot90.html?highlight=rot90#torch.rot90)
```python
torch.rot90(input, k, dims)
```

### 功能介绍
用于实现对矩阵进行k次90度旋转，PaddlePaddle目前无对应API，可使用如下代码组合实现该API。
```python
import paddle
def rot90(input, k, dims):
    l = len(input.shape)
    new_dims = list(range(l))
    new_dims[dims[0]] = dims[1]
    new_dims[dims[1]] = dims[0]
    flip_dim = min(dims)
    for i in range(k):
        input = paddle.transpose(input, new_dims)
        input = paddle.flip(input, [flip_dim])
    return input
```