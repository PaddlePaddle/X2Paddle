
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

### 功能差异：
#### 参数种类：
&#160; &#160; &#160; &#160;tensorflow中，使用placeholder创建一个类型为dtype，形状为shape的输入tensor，对于shape中的batch维度，需要用户使用None指定；
paddlepaddle也有类似参数，但是paddlepaddle默认在第0维为用户插入batch维度，在特殊情形下，用户也可以将append_batch_size设置为False，并使用-1在shape中指定
batch维度所在的位置。paddlepaddle
中的lod_level是paddlepaddle高级特性，普通用户可以暂不理会。


## paddlepaddle示例:
```python

# 创建输入型tensor out，其shape为[-1, 3, 4], 数据类型为float32
out = fluid.layers.data('out', shape=[3, 4], dtype='float32')

# 创建输入型tensor out，其shape为[3, -1, 4], 数据类型为float32
out = fluid.layers.data('out', shape=[3, -1, 4], append_batch_size=False, dtype='float32')
```
