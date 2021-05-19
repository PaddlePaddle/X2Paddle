# torch.nn.Conv1d
### [torch.nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html?highlight=conv1d#torch.nn.Conv1d)

```python
torch.nn.Conv1d(in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros')
```

### [paddle.nn.Conv1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/conv/Conv1D_cn.html#conv1d)

```python
paddle.nn.Conv1D(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCL')
```

### 功能差异

#### 输入格式
***PyTorch***：只支持`NCL`的输入。  
***PaddlePaddle***：支持`NCL`和`NLC`两种格式的输入（通过`data_format`设置）。

#### 更新参数设置
***PyTorch***：`bias`默认为True，表示使用可更新的偏置参数。  
***PaddlePaddle***：`weight_attr`/`bias_attr`默认使用默认的权重/偏置参数属性，否则为指定的权重/偏置参数属性，具体用法参见[ParamAttr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/param_attr/ParamAttr_cn.html#cn-api-fluid-paramattr)；当`bias_attr`设置为bool类型与PyTorch的作用一致。  

#### padding的设置
***PyTorch***：`padding`只能支持list或tuple类型。  
***PaddlePaddle***：`padding`支持list或tuple类型或str类型。
