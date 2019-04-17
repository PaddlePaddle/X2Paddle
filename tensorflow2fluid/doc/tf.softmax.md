## tf.softmax

### [tf.softmax](https://www.tensorflow.org/api_docs/python/tf/nn/softmax)

``` python
tf.nn.softmax(
    logits,
    axis=None,
    name=None,
    dim=None
)
```


### [paddle.fluid.layers.softmax](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#paddle.fluid.layers.softmax)
``` python
paddle.fluid.layers.softmax(
    input, 
    use_cudnn=True, 
    name=None
)
```
### 功能差异

#### 计算方式
TensorFlow: 通过`axis`参数，在指定维度上进行计算，如若`axis`为`None`，则表示在最后一维上进行计算；  
PaddlePaddle：仅在最后一维上进行计算。
