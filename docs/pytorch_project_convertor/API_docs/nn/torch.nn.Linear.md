## torch.nn.Linear
### [torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear)

```python
torch.nn.Linear(in_features, out_features, bias=True)
```

### [paddle.nn.Linear](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Linear_cn.html#linear)

```python
 paddle.nn.Linear(in_features, out_features, weight_attr=None, bias_attr=None, name=None)
```

### 功能差异

#### 更新参数设置
***PyTorch***：`bias`默认为True，表示使用可更新的偏置参数。
***PaddlePaddle***：`weight_attr`/`bias_attr`默认使用默认的权重/偏置参数属性，否则为指定的权重/偏置参数属性，具体用法参见[ParamAttr](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ParamAttr_cn.html#paramattr)；当`bias_attr`设置为bool类型与PyTorch的作用一致。
