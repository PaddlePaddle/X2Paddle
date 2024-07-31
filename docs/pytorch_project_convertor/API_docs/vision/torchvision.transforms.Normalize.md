## torchvision.transforms.Normalize
### [torchvision.transforms.Normalize](https://pytorch.org/vision/stable/transforms.html?highlight=normalize#torchvision.transforms.Normalize)
```python
torchvision.transforms.Normalize(mean, std, inplace=False)
```

### [paddle.vision.transforms.Normalize](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/Normalize_cn.html#normalize)
```python
paddle.vision.transforms.Normalize(mean=0.0, std=1.0, data_format='CHW', to_rgb=False, keys=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| inplace  | -        | 表示表示在不更改变量的内存地址的情况下，直接修改变量，PaddlePaddle无此参数。  |
| -        | data_format      | 表示数据的格式，PyTorch无此参数。                   |
| -        | to_rgb      | 表示是否是否转换为rgb的格式，PyTorch无此参数。                   |

### 功能差异
#### 使用方式
***PyTorch***：只支持`CHW`的输入数据，同时不支持转换为`rgb`。
***PaddlePaddle***：支持`CHW`和`HWC`的输入数据，同时支持转换为`rgb`。
