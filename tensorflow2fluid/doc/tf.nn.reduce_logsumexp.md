## tf.math.reduce_logsumexp

### [tf.math.reduce_logsumexp](https://www.tensorflow.org/api_docs/python/tf/math/reduce_logsumexp)
``` python
tf.math.log_softmax(
    logits,
    axis=None,
    name=None,
    dim=None
)
```

### PaddlePaddle实现
PaddlePaddle中目前无对应接口，可使用如下代码实现
``` python
def reduce_logsumexp(inputs, axis=None, keepdims=None):
    net_0 = fluid.layers.exp(inputs)
    net_1 = fluid.layers.reduce_sum(net_0, dim=axis, keep_dim=keepdims)
    net_2 = fluid.layers.log(net_1)
    return net_2
```

### 代码示例
``` python
inputs = fluid.layers.data(dtype='float32', shape=[1000], name='inputs')

# 调用上述自定义函数
result = reduce_logsumexp(inputs)
```
