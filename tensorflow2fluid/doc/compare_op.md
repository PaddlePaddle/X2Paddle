## 比较函数

在PaddlePaddle中使用运算符来对tensor之间进行`element-wise`方式的对比。其与TensorFlow相应接口关系如下表所示，

| TensorFlow接口 | PaddlePaddle接口 |
|--------------------------|-------------------------------------------------|
|[tf.math.less_equal](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/math/less_equal)|运算符`<=`|
|[tf.math.greater](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/math/greater)|运算符`>`|
|[tf.math.greater_equal](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/math/greater_equal)|运算符`>=`|
|[tf.math.equal](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/math/equal)|运算符`==` 或 [paddle.fluid.layers.equal](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-7-equal) |
|[tf.math.less](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/math/less)|运算符`<` 或 [paddle.fluid.layers.less_than](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-11-less_than) |