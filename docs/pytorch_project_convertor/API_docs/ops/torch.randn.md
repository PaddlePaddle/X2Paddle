### [torch.randn](https://pytorch.org/docs/stable/generated/torch.randn.html?highlight=randn#torch.randn)

```python
torch.randn(*size,
           *,
           out=None,
           dtype=None,
           layout=torch.strided,
           device=None,
           requires_grad=False)
```

### [paddle.randn](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/randn_cn.html#randn)

```python
paddle.randn(shape,
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
torch.randn(2, 3)
# 输出
# tensor([[ 1.3290,  1.4679, -1.2373],
#         [-0.2354, -0.9818,  0.0877]])
```

``` python
# PaddlePaddle示例：
paddle.randn([2, 3])
# 输出
# Tensor(shape=[2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[-1.74181163, -0.50677234, -0.14707172],
#         [ 1.18375409,  1.52477348, -0.73248941]])
```
