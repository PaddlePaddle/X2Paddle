## torch.cholesky_inverse
### [torch.cholesky_inverse](https://pytorch.org/docs/stable/generated/torch.cholesky_inverse.html?highlight=cholesky_inverse#torch.cholesky_inverse)
```python
torch.cholesky_inverse(input, upper=False, out=None)
```

###  功能介绍
用于计算对称正定矩阵的逆矩阵，公式为：   
> 当`upper`为False时，  
> $inv=(uu^T)^{-1}$ ；  
> 当`upper`为True时，  
> $inv=(u^Tu)^{-1}$ 。  


PaddlePaddle目前无对应API，可使用如下代码组合实现该API。

```python
import paddle

def cholesky_inverse(input, upper=False, out=None) :
    u = paddle.cholesky(input, False)
    ut = paddle.transpose(u, perm=[1, 0])
    if upper:
        out = paddle.inverse(paddle.matmul(ut, u))
    else:
        out = paddle.inverse(paddle.matmul(u, ut))
    return out
```