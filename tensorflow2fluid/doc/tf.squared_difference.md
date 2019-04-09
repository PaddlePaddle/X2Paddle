## tf.squared_difference

### [tf.squared_diffenrece](https://www.tensorflow.org/api_docs/python/tf/math/squared_difference)
``` python
tf.math.squared_difference(
    x,
    y,
    name=None
)
```

### PaddlePaddle实现
PaddlePaddle中目前无对应接口，可使用如下代码实现
``` python
def squared_difference(x, y):
    net_0 = fluid.layers.elementwise_sub(x, y)
    net_1 = fluid.layers.elementwise_mul(net_0, net_0)
    return net_1
```

### 代码示例
``` python
input_x = fluid.layers.data(dtype='float32', shape=[1000], name='input_x')
input_y = fluid.layers.data(dtype='float32', shape=[1000], name='input_y')
# 调用上述自定义函数
result = squared_difference(input_x, input_y)
```
