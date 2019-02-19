### tf.get_variable

#### [tf.get_variable](https://www.tensorflow.org/api_docs/python/tf/get_variable)

```python
tf.get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None,
    synchronization=tf.VariableSynchronization.AUTO,
    aggregation=tf.VariableAggregation.NONE
)
```

> tf.get_variable用于创建参数或获取同名参数。Paddle中无完全对应接口，但可通过`create_parameter`创建新的参数，或通过`scope`获取网络中存在的参数  

> [paddle.fluid.layer.create_parameter](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#permalink-201-create_parameter)  

> [如何获取网络参数?](TODO)
