## tf.layers.dense

### [tf.layers.dense](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/layers/dense)
``` python
tf.layers.dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
)
```

### [paddle.fluid.layers.fc](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#fc)
``` python
paddle.fluid.layers.fc(
    input, 
    size, 
    num_flatten_dims=1, 
    param_attr=None, 
    bias_attr=None, 
    act=None, 
    is_test=False, 
    name=None
)

```

### 功能差异
#### 输入类型
TensorFlow：`inputs`为一个tensor；  
PaddlePaddle：允许`input`是一个tensor或者是一个tensor 列表，如果是tensor列表的情况，该layer会声明多个kernel，个数与列表长度相同，在将列表中各个tensor与对应kernel做矩阵乘法之后，将各个结果相加。

#### kernel、bias初始化
TensorFlow：通过`kernel_initializer`与`bias_initializer`对`kernel`、`bias`进行初始化；  
PaddlePaddle：通过设置`param_attr`，`bias_attr`为某种Attribute的方式，进行`kernel`、`bias`初始化。

#### 高维tensor处理
TensorFlow：对于rank大于2的输入tensor，将其看做是最内两个维度所组成矩阵的堆叠，dense操作将改变最后一个维度；  
PaddlePaddle：对于rank大于2的输入tensor，可以从第`num_flatten_dims`维开始（维度下标从0开始，`num_flatten_dims`最小为1），将各维度拍平，例如`shape`为（2，3，4，5），当`num_flatten_dims`为2时，输入tensor将被reshape成(2，3，20)的tensor，输出tensor的shape为(2，3，size)。

### 代码示例
```python
# 输入 tensor t 的shape为[2, 3, 4, 5]

# size=6, 输出tensor 的shape为[2,6] 
out = fluid.layers.fc(t, size=6)

# size=6, 设置kernel为均匀分布
out = fluid.layers.fc(t, size=6, \
    param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(low=-0.5, high=0.5)))

# size=6, num_flatten_dims=2，输出tensor的shape为[2, 3, 6]
out = fluid.layers.fc(t, size=6, num_flatten_dims=2)

```