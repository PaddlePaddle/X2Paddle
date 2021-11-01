## torch.split
### [torch.split](https://pytorch.org/docs/stable/generated/torch.split.html?highlight=torch%20split#torch.split)

```python
torch.split(tensor,
            split_size_or_sections,
            dim=0)
```

### [paddle.split](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/split_cn.html#split)

```python
paddle.split(x,
             num_or_sections,
             axis=0,
             name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | x        | 表示输入Tensor。                                     |
| split_size_or_sections           | num_or_sections            | 当类型为int时，torch表示单个块大小，paddle表示结果有多少个块 |
| dim        | axis            | 表示需要分割的维度。                   |


### 功能差异

#### 使用方式
***PyTorch***：第二个参数split_size_or_sections类型为int或者list(int)。  
***PaddlePaddle***：第二个参数num_or_sections类型为int、list(int)或者tuple(int)。


### 代码示例
``` python
# PyTorch示例：
a = torch.arange(10).reshape(5,2)
# 输出
# tensor([[0, 1],
#         [2, 3],
#         [4, 5],
#         [6, 7],
#         [8, 9]])
torch.split(a, 2, 1)
# 输出
# (tensor([[0, 1],
#         [2, 3],
#         [4, 5],
#         [6, 7],
#         [8, 9]]),)
```

``` python
# PaddlePaddle示例：
b = paddle.arange(10).reshape([5,2])
# 输出
# Tensor(shape=[5, 2], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [[0, 1],
#         [2, 3],
#         [4, 5],
#         [6, 7],
#         [8, 9]])
paddle.split(b, 2, 1)
# 输出
# [Tensor(shape=[5, 1], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [[0],
#         [2],
#         [4],
#         [6],
#         [8]]), Tensor(shape=[5, 1], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [[1],
#         [3],
#         [5],
#         [7],
#         [9]])]
```
