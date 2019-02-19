
## tf.constant

### [tf.constant](https://www.tensorflow.org/api_docs/python/tf/constant)
``` python
tf.constant(
    value,
    dtype=None,
    shape=None,
    name='Const',
    verify_shape=False
)
```

### [paddle.fluid.layers.fill_constant](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#cn-api-fluid-layers-fill-constant)
``` python
paddle.fluid.layers.fill_constant(
    shape, 
    dtype, 
    value, 
    force_cpu=False, 
    out=None
)
```

### 功能差异：
#### 参数类型差异：
>  tensorflow：value可以是scalar或者是python list，shape是可选的，在value.shape与shape不兼容情况下，将使用value的最后element做填充。
>  paddlepaddle：value必须是scalar，根据shape来生成constant tensor。


## paddlepaddle示例:
```python

# 常量tensor out 中数据为 np.array([[5,5,5],[5,5,5]], dtype='int64')
out = fluid.layers.fill_constant(shape=[2,3], dtype='int64', value=5)  
