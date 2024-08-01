## torch.nn.BatchNorm3d
### [torch.nn.BatchNorm3d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html?highlight=torch%20nn%20batchnorm3d#torch.nn.BatchNorm3d)
```python
torch.nn.BatchNorm3d(num_features,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True)
```
### [paddle.nn.BatchNorm3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/BatchNorm3D_cn.html#batchnorm3d)
```python
paddle.nn.BatchNorm3D(num_features,
                      momentum=0.9,
                      epsilon=1e-05,
                      weight_attr=None,
                      bias_attr=None,
                      data_format='NCDHW',
                      use_global_stats=True,
                      name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| eps          | epsilon        | 为了数值稳定加在分母上的值。                                     |
| -           | weight_attr            | 指定权重参数属性的对象。如果为False, 则表示每个通道的伸缩固定为1，不可改变。默认值为None，表示使用默认的权重参数属性。 |
| -           | bias_attr            | 指定偏置参数属性的对象。如果为False, 则表示每一个通道的偏移固定为0，不可改变。默认值为None，表示使用默认的偏置参数属性。 |
| affine  | -   | 是否进行反射变换，PaddlePaddle无此参数。         |
| track_running_stats  | use_global_stats   | 表示是否已加载的全局均值和方差。                   |

### 功能差异

#### 输入格式
***PyTorch***：只支持`NCDHW`的输入。
***PaddlePaddle***：支持`NCDHW`和`NDHWC`两种格式的输入（通过`data_format`设置）。

#### 反射变换设置
当PyTorch的反射变换设置为`False`，表示weight和bias不进行更新，由于PaddlePaddle不具备这一功能，可使用如下代码组合实现该API。
```python
class BatchNorm3D(paddle.nn.BatchNorm3D):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        momentum = 1 - momentum
        weight_attr = None
        bias_attr = None
        if not affine:
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super().__init__(
            num_features,
            momentum=momentum,
            epsilon=eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=track_running_stats)
```
