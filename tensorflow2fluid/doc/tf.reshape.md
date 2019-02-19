
## tf.reshape

### [tf.reshape](https://www.tensorflow.org/api_docs/python/tf/reshape)
``` python
tf.reshape(
    tensor,
    shape,
    name=None
)
```

### [paddle.fluid.layers.unsqueeze](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#cn-api-fluid-layers-reshape)
``` python
paddle.fluid.layers.reshape(
    x, 
    shape, 
    actual_shape=None, 
    act=None, 
    inplace=False, 
    name=None)
```

### 功能差异：
#### 参数类型差异：
&#160; &#160; &#160; &#160;tensorflow中，shape 可以是python list，也可以是变量类型；paddlepaddle中，shape 只能是python list。可选参数actual_shape支持变量类型，其优先级高于shape。需要注意的是，在设置actual_shape的时候，也要正确设置shape以便正常通过paddle编译阶段。

#### shape标记差别：
&#160; &#160; &#160; &#160;tensorflow中，shape 中可以使用单独一个-1，表示待推断的维度；paddlepaddle中，shape 中除了可以使用单独一个-1表示待推断维度外，还能使用0，表示在输入tensor原来的shape中对应位置的维度。注意，0的下标不能超过原来tensor的rank。


## paddlepaddle示例:
```python
# 输入 tensor t 的 shape 为[3, 4]

# 输出 tensor out 的 shape 为[2，6]
out = fluid.layers.reshape(t, [-1, 6])  

# 输出 tensor out 的 shape 为[3, 2, 2]
out = fluid.layers.reshape(t, [0, 2, 2])
```

