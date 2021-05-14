## torch.nn.Dropout3d
### [torch.nn.Dropout3d](https://pytorch.org/docs/stable/generated/torch.nn.Dropout3d.html?highlight=dropout3d#torch.nn.Dropout3d)
```python
torch.nn.Dropout3d(p=0.5, inplace=False)
```
### [paddle.nn.Dropout3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/common/Dropout3D_cn.html#dropout3d)
```python
paddle.nn.Dropout3D(p=0.5, data_format='NCDHW', name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| inplace          | -        | 表示在不更改变量的内存地址的情况下，直接修改变量的值，PaddlePaddle无此参数。  |
| -           | data_format            | 指定对输入的数据格式，PyTorch无此参数。 |

### 功能差异

#### 输入格式
***PyTorch***：只支持`NCDHW`的输入。  
***PaddlePaddle***：支持`NCDHW`和`NDHWC`两种格式的输入（通过`data_format`设置）。
