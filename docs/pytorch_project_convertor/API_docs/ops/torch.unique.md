## torch.unique
### [torch.unique](https://pytorch.org/docs/stable/generated/torch.unique.html?highlight=unique#torch.unique)
```python
torch.unique(input, sorted, return_inverse, return_counts, dim=None)
```
### [paddle.unique](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/unique_cn.html#unique)
```python
paddle.unique(x, return_index=False, return_inverse=False, return_counts=False, axis=None, dtype='int64', name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input | x        | 表示输入Tensor。  |
| sorted          | -        | 表示升序或者降序，PaddlePaddle无此参数。  |
| -          | return_index        | 表示是否返回独有元素在输入Tensor中的索引，PyTorch无此参数。  |
| return_inverse          | return_inverse        | 表示是否返回输入Tensor的元素对应在独有元素中的索引。  |
| return_counts          | return_counts        | 表示是否返回每个独有元素在输入Tensor中的个数。  |
| dim          | axis        | 表示指定选取独有元素的轴。  |
| dtype          | -        | 表示返回值的类型，PyTorch无此参数。  |

### 功能差异
#### 返回值差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| output | out        | 表示独有元素构成的Tensor。  |
| inverse_indices      | inverse        | （可选）表示输入Tensor的元素对应在独有元素中的索引。  |
| counts         | counts        | （可选）表示每个独有元素在输入Tensor中的个数。  |
| -          | index        | （可选）表示独有元素在输入Tensor中的索引，PyTorch无此返回值。  |

***PyTorch***：（1）可以通过`sorted`设置是返回升序还是降序结果；（2）`inverse_indices`形状与原形状相同。
***PaddlePaddle***：（1）只能返回升序结果；（2）`inverse`为`inverse_indices`展平后形状。

### 代码示例
``` python
# PyTorch示例：
import torch
x = torch.tensor([[[1, 3], [2, 3]], [[1, 6], [2, 3]]], dtype=torch.float32)
output, inverse_indices, counts = torch.unique(x, 
                      sorted=True, 
                      return_inverse=True,
                      return_counts=True)
# 输出
# output:
# tensor([1., 2., 3., 6.])
# inverse_indices:
# tensor([[[0, 2],
#          [1, 2]],

#         [[0, 3],
#          [1, 2]]])
# counts:
# tensor([2, 2, 3, 1])
```

``` python
# PaddlePaddle示例：
import paddle
x = paddle.to_tensor([[[1, 3], [2, 3]], [[1, 6], [2, 3]]], dtype="float32")
out, inverse, counts = paddle.unique(x, 
                      return_inverse=True,
                      return_counts=True)
# 输出
# out:
# Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [1., 2., 3., 6.])
# inverse:
# Tensor(shape=[8], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [0, 2, 1, 2, 0, 3, 1, 2])
# counts:
# Tensor(shape=[4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [2, 2, 3, 1])
```
