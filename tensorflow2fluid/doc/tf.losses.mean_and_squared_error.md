## tf.losses.mean_and_squared_error

### [tf.losses.mean_and_squared_error](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/losses/mean_squared_error)

``` python
tf.losses.mean_squared_error(
    labels,
    predictions,
    weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
)
```


### [paddle.fluid.layers.square_error_cost](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#paddle.fluid.layers.square_error_cost)
``` python
paddle.fluid.layers.square_error_cost(
    input, 
    label
)
```
### 功能差异

#### 计算方式
TensorFlow: 提供`weights`参数，通过传入`weights`参数的shape，可实现不同的加权方式；  
PaddlePaddle：不支持加权。