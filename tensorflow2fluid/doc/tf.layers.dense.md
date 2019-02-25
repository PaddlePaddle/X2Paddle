
### tf.layers.dense

#### [tf.layers.dense](https://www.tensorflow.org/api_docs/python/tf/layers/dense)
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

#### [paddle.fluid.layers.fc](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#fc)
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

#### 功能差异：
##### 输入类型
tensorflow 中inputs 为一个tensor；paddlepaddle中允许是一个tensor或者是一个tensor 列表，如果是tensor列表的情况，该layer会声明多个kernel，个数与列表长度相同，在将列表中各个tensor与对应kernel做矩阵乘法之后，将各个结果相加。

##### kernel、bias初始化
tensorflow 中通过kernel_initializer与bias_initializer对kernel、bias进行初始化；paddlepaddle中通过设置param_attr，bias_attr为某种Attribute的方式，进行kernel、bias初始化。

##### 高维tensor处理
tensorflow 中，对于rank大于2的输入tensor，将其看做是最内两个维度所组成矩阵的堆叠，dense操作将改变最后一个维度；paddlepaddle中，对于rank大于2的输入tensor，可以从第num_flatten_dims维开始（维度下标从0开始，num_flatten_dims最小为1），将各维度拍平，例如shape为（2，3，4，5），当num_flatten_dims=2时，输入tensor将被reshape成(2，3，20)的tensor，输出tensor的shape为（2，3，size）

#### paddlepaddle示例:
```python
# 输入 tensor t 的shape为[2, 3, 4, 5]

# size=6, 输出tensor 的shape为[2,6] 
out = fluid.layers.fc(t, size=6)

# # size=6, 设置kernel为均匀分布
out = fluid.layers.fc(t, size=6, \
    param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(low=-0.5, high=0.5)))

# size=6, num_flatten_dims=2，输出tensor的shape为[2, 3, 6]
out = fluid.layers.fc(t, size=6, num_flatten_dims=2)

```
