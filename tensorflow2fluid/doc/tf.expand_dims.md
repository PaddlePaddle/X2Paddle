
## tf.expand_dims

### [tf.expand_dims](https://www.tensorflow.org/api_docs/python/tf/expand_dims)
``` python
tf.expand_dims(
    input,
    axis=None,
    name=None,
    dim=None
)
```

### [paddle.fluid.layers.unsqueeze](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#unsqueeze)
``` python
paddle.fluid.layers.unsqueeze(
    input, 
    axes, 
    name=None)
```

### 功能差异

#### 参数类型
TensorFlow：`axis`为`int`类型或`0-D`tensor, 使用`axis`指定要增加维度的位置，支持负数进行索引；

PaddlePaddle：`axes`为`list`类型，表示要增加维度的位置列表，支持在多个位置同时增加维度，也支持负数进行索引。


### 代码示例
```python
# 输入 tensor t 的 shape 为[3, 4]

# 输出 tensor out 的 shape 为[1, 3, 4]
out = fluid.layers.unsqueeze(t, [0])  

# 输出 tensor out 的 shape 为[3, 4, 1]
out = fluid.layers.unsqueeze(t, [-1])

# 输出 tensor out 的 shape 为[1, 1，3, 4]
out = fluid.layers.unsqueeze(t, [0, 1])  
```

