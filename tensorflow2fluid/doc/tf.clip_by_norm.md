## tf.clip_by_norm

### [tf.clip_by_norm](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/clip_by_norm)

``` python
tf.clip_by_norm(
    t,
    clip_norm,
    axes=None,
    name=None
)
```


### [paddle.fluid.layers.clip_by_norm](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#paddle.fluid.layers.clip_by_norm)
``` python
paddle.fluid.layers.clip_by_norm(
    x, 
    max_norm, 
    name=None
)
```
### 功能差异

#### 计算方式
TensorFlow: 使用参数`axis`指定的轴计算L2范数`l2-norm`，如若`axis`为None，则表示使用整个输入数据的L2范数；  
PaddlePaddle：使用整个输入数据的L2范数。