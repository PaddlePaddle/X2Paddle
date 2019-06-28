## tf.reshape

### [tf.reshape](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/reshape)
``` python
tf.reshape(
    tensor,
    shape,
    name=None
)
```

### [paddle.fluid.layers.reshape](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#cn-api-fluid-layers-reshape)
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

#### shape标记差别
TensorFlow: shape 中可以使用单独一个-1，表示待推断的维度；  
PaddlePaddle: shape 中除了可以使用单独一个-1表示待推断维度外，还能使用0，表示在输入tensor原来的shape中对应位置的维度。注意，0的下标不能超过原来tensor的rank。


## 代码示例
```python
# 输入 tensor t 的 shape 为[3, 4]

# 输出 tensor out 的 shape 为[2，6]
out = fluid.layers.reshape(t, [-1, 6])  

# 输出 tensor out 的 shape 为[3, 2, 2]
out = fluid.layers.reshape(t, [0, 2, 2])
```