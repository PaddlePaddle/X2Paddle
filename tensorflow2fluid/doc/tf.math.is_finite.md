
## tf.math.is_finite

### [tf.math.is_finite](https://www.tensorflow.org/api_docs/python/tf/math/is_finite)
``` python
tf.math.is_finite(
    x,
    name=None
)
```

### [paddle.fluid.layers.isfinite](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#paddle.fluid.layers.isfinite)
``` python
paddle.fluid.layers.isfinite(x)
```

### 功能差异

#### 输出格式
TensorFlow: 返回elementwise检查的结果，即输出与输入shape一致；  
PaddlePaddle: 返回结果仅包含一个boolean值，若输入数据中均为`infinite`，则返回True，否则返回False。

### 代码示例
```python
# TensorFlow示例
# 输入[2.1, 3.2, 4.5]
# 输出[True, True, True]
result = tf.is_finite(inputs)

# PaddlePaddle示例
# 输入[2.1, 3.2, 4.5]
# 输出True
result = fluid.layers.isfinite(inputs)
```
