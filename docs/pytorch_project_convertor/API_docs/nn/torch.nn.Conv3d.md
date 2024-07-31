# torch.nn.Conv3d
### [torch.nn.Conv3d](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html?highlight=conv3d#torch.nn.Conv3d)

```python
torch.nn.Conv3d(in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros')
```

### [paddle.nn.Conv3D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Conv3D_cn.html#conv3d)

```python
paddle.nn.Conv3D(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCDHW')
```

### 功能差异

#### 输入格式
***PyTorch***：只支持`NCHW`的输入。
***PaddlePaddle***：支持`NCDHW`和`NDHWC`两种格式的输入（通过`data_format`设置）。

#### 更新参数设置
***PyTorch***：`bias`默认为True，表示使用可更新的偏置参数。
***PaddlePaddle***：`weight_attr`/`bias_attr`默认使用默认的权重/偏置参数属性，否则为指定的权重/偏置参数属性，具体用法参见[ParamAttr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ParamAttr_cn.html#paramattr)；当`bias_attr`设置为bool类型与PyTorch的作用一致。
#### padding的设置
***PyTorch***：`padding`只能支持list或tuple类型。它可以有3种格式：
(1)包含4个二元组：\[\[0,0\], \[0,0\], \[padding_depth_front, padding_depth_back\], \[padding_height_top, padding_height_bottom\], \[padding_width_left, padding_width_right\]\]，其中每个元组都可使用整数值替换，代表元组中的2个值相等；
(2)包含3个二元组：\[\[padding_depth_front, padding_depth_back\], \[padding_height_top, padding_height_bottom\], \[padding_width_left, padding_width_right\]\]，其中每个元组都可使用整数值替换，代表元组中的2个值相等；
(3)包含一个整数值，padding_height = padding_width = padding。
***PaddlePaddle***：`padding`支持list或tuple类型或str类型。如果它是一个list或tuple，它可以有4种格式：
(1)包含5个二元组：当 data_format 为"NCDHW"时为 \[\[0,0], \[0,0\], \[padding_depth_front, padding_depth_back\], \[padding_height_top, padding_height_bottom\], \[padding_width_left, padding_width_right\]\]，当 data_format 为"NDHWC"时为\[\[0,0\], \[padding_depth_front, padding_depth_back\], \[padding_height_top, padding_height_bottom\], \[padding_width_left, padding_width_right\], \[0,0\]\]；
(2)包含6个整数值：\[padding_depth_front, padding_depth_back, padding_height_top, padding_height_bottom, padding_width_left, padding_width_right\]；
(3)包含3个整数值：\[padding_depth, padding_height, padding_width\]，此时 padding_depth_front = padding_depth_back = padding_depth, padding_height_top = padding_height_bottom = padding_height, padding_width_left = padding_width_right = padding_width；
(4)包含一个整数值，padding_height = padding_width = padding。如果它为一个字符串时，可以是"VALID"或者"SAME"，表示填充算法。
