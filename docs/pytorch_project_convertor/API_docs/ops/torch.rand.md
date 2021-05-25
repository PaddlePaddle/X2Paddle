### [torch.rand](https://pytorch.org/docs/stable/generated/torch.rand.html?highlight=rand#torch.rand)

```python
torch.rand(*size,
           *,
           out=None,
           dtype=None,
           layout=torch.strided,
           device=None,
           requires_grad=False)
```

### [paddle.rand](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/rand_cn.html#rand)

```python
paddle.rand(shape,
             dtype=None,
             name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size          | shape        | 表示输出形状大小。                                     |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |
| layout        | -            | 表示布局方式，PaddlePaddle无此参数。                   |
| device        | -            | 表示Tensor存放位置，PaddlePaddle无此参数。                   |
| requires_grad | -            | 表示是否不阻断梯度传导，PaddlePaddle无此参数。 |

### 功能差异

#### 使用方式
***PyTorch***：生成Tensor的形状大小以可变参数的方式传入。  
***PaddlePaddle***：生成Tensor的形状大小以list或tuple的方式传入。


### 代码示例
``` python
# PyTorch示例：
torch.rand(2, 3)
# 输出
# tensor([[0.0860, 0.2757, 0.3211],
#         [0.5872, 0.5267, 0.4184]])
```

``` python
# PaddlePaddle示例：
paddle.rand([2, 3])
# 输出
# Tensor(shape=[2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[0.18905126, 0.56219709, 0.00808361],
#         [0.78120756, 0.32112977, 0.90572405]])
```
