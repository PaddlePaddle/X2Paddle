
## tf.placeholder

### [tf.placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
``` python
tf.placeholder(
    dtype,
    shape=None,
    name=None
)
```

### [paddle.fluid.layers.data](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#cn-api-fluid-layers-data)
``` python
paddle.fluid.layers.data(
    name, 
    shape, 
    append_batch_size=True, 
    dtype='float32', 
    lod_level=0, 
    type=VarType.LOD_TENSOR, 
    stop_gradient=True)
```

### 功能差异
#### Batch维度处理
TensorFlow: 对于shape中的batch维度，需要用户使用`None`指定；  
PaddlePaddle: 将第1维设置为`-1`表示batch维度；如若第1维为正数，则会默认在最前面插入batch维度，如若要避免batch维，可将参数`append_batch_size`设为`False`。


### 代码示例
```python

# 创建输入型tensor out，其shape为[-1, 3, 4], 数据类型为float32
out = fluid.layers.data('out', shape=[3, 4], dtype='float32')

# 创建输入型tensor out，其shape为[3, -1, 4], 数据类型为float32
out = fluid.layers.data('out', shape=[3, -1, 4], append_batch_size=False, dtype='float32')
```
