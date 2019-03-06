
## tf.constant

### [tf.random_uniform](https://www.tensorflow.org/api_docs/python/tf/random/uniform)
``` python
tf.random.uniform(
    shape,
    minval=0,
    maxval=None,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
```

### [paddle.fluid.layers.uniform_random](http://paddlepaddle.org/documentation/docs/en/1.3/api/layers.html#permalink-203-uniform_random)
``` python
paddle.fluid.layers.uniform_random(
    shape, 
    dtype=None, 
    min=None, 
    max=None, 
    seed=None)
```

### 功能差异

#### 默认参数
Tensorflow: 默认数值范围`[0.0, 1.0)`  
PaddlePaddle: 默认数值范围`[-1.0, 1.0)`

#### 其它
如若使用PaddlePaddle的uniform_random，需要生成带`batch_size`维度的`tensor`，需要使用[uniform_random_batch_size_like](http://paddlepaddle.org/documentation/docs/en/1.3/api/layers.html#permalink-178-uniform_random_batch_size_like)
