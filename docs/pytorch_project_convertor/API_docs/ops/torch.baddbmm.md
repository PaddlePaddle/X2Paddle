## torch.baddbmm
### [torch.baddbmm](https://pytorch.org/docs/stable/generated/torch.baddbmm.html?highlight=baddbmm#torch.baddbmm)
```python
torch.baddbmm(input, batch1, batch2, beta=1, alpha=1, out=None)
```
###  功能介绍
用于实现Tensor（大小为$b×n×m$）与用于实现Tensor（大小为$b×m×p$）相乘，再加上输入（`input`），公式为：  
$out_i = β *  input_i + α * (batch1_i @ batch2_i)$  
PaddlePaddle目前无对应API，可使用如下代码组合实现该API。

```python
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