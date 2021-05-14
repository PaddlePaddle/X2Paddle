## torch.nn.MaxPool3d
### [torch.nn.MaxPool3d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html?highlight=maxpool3d#torch.nn.MaxPool3d)

```python
torch.nn.MaxPool3d(kernel_size,
                   stride=None,
                   padding=0,
                   dilation=1,
                   return_indices=False,
                   ceil_mode=False)
```

### [paddle.nn.MaxPool3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/pooling/MaxPool3D_cn.html#maxpool3d)

```python
paddle.nn.MaxPool3D(kernel_size,
                    stride=None,
                    padding=0,
                    ceil_mode=False,
                    return_mask=False,
                    data_format='NCDHW',
                    name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dilation           | -            | 设置空洞池化的大小，PaddlePaddle无此参数。               |

### 功能差异

#### 池化方式
***PyTorch***：可以使用空洞池化。  
***PaddlePaddle***：无此池化方式。
