## tf.math.rsqrt

### [tf.math.rsqrt](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/math/rsqrt)
``` python
tf.math.rsqrt(
    x,
    name=None
)
```

### PaddlePaddle实现
PaddlePaddle中目前无对应接口，可使用如下代码实现
``` python
def rsqrt(x):
    net_0 = fluid.layers.sqrt(x)
    net_1 = fluid.layers.pow(net_0, factor=-1.0)
    return net_1
```

### 代码示例
``` python
inputs = fluid.layers.data(dtype='float32', shape=[1000], name='inputs')

# 调用上述自定义函数
result = rsqrt(inputs)
```