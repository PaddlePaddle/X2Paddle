## torch.mv
### [torch.mv](https://pytorch.org/docs/stable/generated/torch.mv.html?highlight=mv#torch.mv)
```python
torch.mv(input, vec, out=None)
```

###  功能介绍
用于实现矩阵（`input`，大小为$n × m$）与向量（`vec`, $m$）相乘，PaddlePaddle目前无对应API，可使用如下代码组合实现该API。
```python
import paddle

def mv(input, vec, out=None):
    vec = paddle.unsqueeze(vec, 1)
    out = paddle.matmul(input, vec)
    out = paddle.squeeze(out, 1)
    return out
```