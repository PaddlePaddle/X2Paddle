## torch.chain_matmul
### [torch.chain_matmul](https://pytorch.org/docs/stable/generated/torch.chain_matmul.html?highlight=chain_matmul#torch.chain_matmul)
```python
torch.chain_matmul(*matrices, out=None)
```
###  功能介绍
用于实现多个矩阵相乘，PaddlePaddle目前无对应API，可使用如下代码组合实现该API。
```python
import paddle

def chain_matmul(*matrices, out=None):
    assert len(matrices) >= 1, "Expected one or more matrices."
    if len(matrices) == 1:
        return matrices[0]
    out = paddle.matmul(matrices[0], matrices[1])
    for i in range(2, len(matrices)):
        out = paddle.matmul(out, matrices[i])
    return out
```