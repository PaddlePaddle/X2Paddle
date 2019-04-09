
## tf.one_hot

### [tf.one_hot](https://www.tensorflow.org/api_docs/python/tf/one_hot)
``` python
tf.one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)
```

### [paddle.fluid.layers.one_hot](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#one-hot)
``` python
layers.one_hot(;
    input, 
    depth
)
```

### 功能差异
#### 输入格式
TensorFlow：indices shape 没有限定；支持设置on与off的值；

PaddlePaddle：input限定为2-D tensor，shape为(batch, 1)。

#### 参数种类
TensorFlow：可以配置`on_value`和`off_value`，默认为`1`和`0`；  
PaddlePaddle：无对应配置选项，即为默认的`1`和`0`。

### 代码示例
```python
# 输入 tensor t 为[[1],[2]]

# depth 为3时，输出 tensor out 为[[0, 1, 0], [0, 0, 1]]
out = fluid.layers.one_hot(t, 3)
```
