
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
##### 输入类型：
tensorflow 中inputs 为一个tensor；paddlepaddle中允许是一个tensor或者是一个tensor 列表，如果是tensor列表的情况，该layer会声明多个kernel，个数与列表长度相同，在将列表中各个tensor与对应kernel做矩阵乘法之后，将各个结果相加。

##### 

tensorflow：使用rate表示保留为0的概率；使用noise_shape来设置dropout的独立性；  
paddlepaddle：使用dropout_prob表示保留为0的概率；通过设置is_test来打开或关闭dropout，通常用户可以结合default_main_program的clone方法，来设置is_test的值；不支持dropout独立性设置；另外通过设置dropout_implementation，还可以设定dropout的具体实现方式：当设定为"upscale_in_train"时，被保留的元素会乘上一个scale系数，保持输出与输入的所有元素均值一致，该方式与tensorflow的实现一致。

#### paddlepaddle示例:
```python
# 输入 tensor t 为[[1,2],[3,4]]

# 第0维前面padding长度为0，后面padding长度为1；第1维前面padding长度为1，后面padding长度为2
out = fluid.layers.dropout(t, dropout_prob=0.2, dropout_implementation="upscale_in_train")

# inference 时关闭dropout
inference_program = fluid.default_main_program().clone(for_test=True)
```
