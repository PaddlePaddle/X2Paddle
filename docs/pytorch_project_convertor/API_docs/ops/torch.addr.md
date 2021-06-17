## torch.addr
### [torch.addr](https://pytorch.org/docs/stable/generated/torch.addr.html?highlight=addr#torch.addr)
```python
torch.addr(input, vec1, vec2, beta=1, alpha=1, out=None)
```

###  功能介绍
用于实现矩阵（`vec`）与向量（`vec`）相乘，再加上输入（`input`），公式为：  
$out = β * input + α *  (vec1 ⊗ vec2)$  
PaddlePaddle目前无对应API，可使用如下代码组合实现该API。

```python
import paddle

def addr(input, vec1, vec2, beta=1, alpha=1, out=None):
    row = vec1.shape[0]
    column = vec2.shape[0]
    vec1 = paddle.unsqueeze(vec1, 0)
    vec1 = paddle.transpose(vec1, [1, 0])
    vec1 = paddle.expand(vec1, [row, column])
    new_vec2 = paddle.zeros([column, column], dtype=vec2.dtype)
    new_vec2[0, :] = vec2
    out = alpha * paddle.matmul(vec1, new_vec2)
    out = beta * input + out
    return out
```