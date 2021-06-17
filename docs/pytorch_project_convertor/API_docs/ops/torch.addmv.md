## torch.addmv
### [torch.addmv](https://pytorch.org/docs/stable/generated/torch.addmv.html?highlight=addmv#torch.addmv)
```python
torch.addmv(input, mat, vec, beta=1, alpha=1, out=None)
```

###  功能介绍
用于实现矩阵（`mat`）与向量（`vec`）相乘，再加上输入（`input`），公式为：  
$ out = β *  input + α *  (mat @ vec) $  
PaddlePaddle目前无对应API，可使用如下代码组合实现该API。

```python
import paddle

def addmv(input, mat, vec, beta=1, alpha=1, out=None):
    mv = alpha * paddle.matmul(mat, vec)
    input = beta * input
    out = mv + input
    return out
```