## torch.nn.Dropout
### [torch.nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html?highlight=dropout#torch.nn.Dropout)
```python
torch.nn.Dropout(p=0.5, inplace=False)
```

### [paddle.nn.Dropout](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Dropout_cn.html#dropout)
```python
paddle.nn.Dropout(p=0.5, axis=None, mode="upscale_in_train”, name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| inplace          | -        | 表示在不更改变量的内存地址的情况下，直接修改变量的值，PaddlePaddle无此参数。  |
| -           | axis            | 指定对输入Tensor进行Dropout操作的轴，PyTorch无此参数。 |
| -           | mode            | 表示丢弃单元的方式，PyTorch无此参数。|


### 功能差异

#### 丢弃方式
***PyTorch***：只支持`upscale_in_train`的丢弃方式。  
***PaddlePaddle***：支持`upscale_in_train`和`downscale_in_infer`两种丢弃方式（通过`mode`设置），计算方法可参考[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/common/Dropout_cn.html#dropout)。
