## CV常用API

### tf.layers.conv2d

#### [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d)

```python
tf.layers.conv2d(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format='channels_last',
    dilation_rate=(1, 1),
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

#### [paddle.fluid.layers.conv2d](http://www.paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#paddle.fluid.layers.conv2d)

```python
paddle.fluid.layers.conv2d(
    input,
    num_filters,
    filter_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=None,
    param_attr=None,
    bias_attr=None,
    use_cudnn=True,
    act=None,
    name=None
)
```


#### 功能差异

1. 输入数据格式
2. padding机制

##### 1. 数据格式差异

&#160; &#160; &#160; &#160;TensorFlow中，通过`data_format`参数，可以指定输入的图像数据格式为`NHWC`或`NCHW`，默认条件下，同时也是目前多数tensorflow模型采用的输入数据格式都为`NHWC`；而PaddlePaddle在目前版本中，conv2d的输入固定数据格式为`NCHW`。通过下表，列出了两者在conv2d上的数据格式差异。

conv2d|input|Kernel|output
:-----------:|:--------------------------:|:---------------------------:|:-----------------------------:
TensorFlow|(batch, height, width, channel)|(height, width, channel, filter_num)|(batch, height, width, filter_num)
PaddlePaddle|(batch, channel, height, width)|(filter_num, channel, height, width)|(batch, filter_num, height, width)

##### 2. padding差异

&#160; &#160; &#160; &#160;TensorFlow中，conv2d包括`VALID`和`SAME`两种padding机制，其中前者对输入数据不做padding处理，而后者在padding时，padding所需的size计算方式为`pad_size = (ceil(input_size / stride) - 1) * stride + filter_size - input_size`，pad_size会分成两份分别pad在数据的两端，且当计算得到的pad_size为奇数时，right/bottom则会相对多pad一个size。

&#160; &#160; &#160; &#160;PaddlePaddle中，conv2d的padding参数为整型的值或者tuple，表示(padding_height, padding_width), 分别在输入图像数据的上下左右padding, 例如在输入(3, 200, 200)的图像数据，经过(1, 1)的padding后，得到的数据shape为(3, 202, 202)。在使用conv2d前，也可以调用`fluid.layers.pad2d`对输入数据做更复杂的padding操作。

&#160; &#160; &#160; &#160;如下示例展示了PaddlePaddle和Tensorflow的对应代码实现

Tensorflow  
```
# 输入shape：(None, 200, 200, 3)
# 输出shape：(None, 67， 67， 5）
inputs = tf.placeholder(tf.float32, shape=[None, 200, 200, 3], name='inputs')
outputs = tf.layers.conv2d(inputs, 5, [5, 5], (1, 1), 'SAME')
```
PaddlePaddle
```
# 输入Shape：(None, 3, 200, 200)
# 输出Shape：(None, 5， 67， 67）
inputs = paddle.fluid.layers.data(dtype='float32', shape=[3, 200, 200], name='inputs)
pad_inputs = paddle.fluid.layers.pad2d(inputs, paddings=[1, 2, 1, 2])
outputs = paddle.fluid.layers.conv2d(pad_inputs, 5, [4, 4], (1, 1))
```

##### 3. 其它差异

&#160; &#160; &#160; &#160;在PaddlePaddle的conv2d接口中，通过`group`参数的设定，可实现对应tensorflow的depthwise卷积和pointwise卷积，参见文档[tf.layers.separable_conv2d](www.baidu.com)


## NLP常用API

### tf.nn.dynamic_rnn

#### [tf.nn.dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
``` python
tf.nn.dynamic_rnn(
    cell,
    inputs,
    sequence_length=None,
    initial_state=None,
    dtype=None,
    parallel_iterations=None,
    swap_memory=False,
    time_major=False,
    scope=None
)
```

#### [paddle.fluid.layers.DynamicRNN](http://www.paddlepaddle.org/documentation/docs/zh/1.2/api_cn/api_guides/low_level/layers/control_flow.html#dynamicrnn)
``` python
paddle.fluid.layers.DynamicRNN(name=None)
```

#### 功能差异
##### 1. 调用机制差异
&#160; &#160; &#160; &#160;Tensorflow中，`tf.nn.dynamic_rnn`通常与`tf.nn.rnn_cell.LSTMCell`、`tf.nn.rnn_cell.GRUCell`等Cell结合使用；而在paddlepaddle中，使用`paddle.fluid.layers.DynamicRNN`类实现类似功能 ，通过DynamicRNN提供的类方法，用户可以在`with block`中方便地自定义每个时间步的处理过程。

##### 2. 输入差异
&#160; &#160; &#160; &#160;Tensorflow中，tf.nn.dynamic_rnn为序列数据，批输入中的每个序列需要填充到相同的长度；而在paddlepaddle中，使用
[LoDTensor](http://www.paddlepaddle.org/documentation/docs/zh/1.2/user_guides/howto/basic_concept/lod_tensor.html)表示一个批输入，用户在使用时不需要进行填充操作。

Tensorflow
```
# 创建 BasicRNNCell
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

# 定义初始隐状态
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

# 输出shape为（batch_size, max_time, cell_state_size）
# 最后时刻隐状态shape为（batch_size, cell_state_size）
outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                   initial_state=initial_state,
                                   dtype=tf.float32)
```

paddlepaddle
```
# 创建一个DynamicRNN对象
drnn = fluid.layers.DynamicRNN()

# 定义一个类似BasicRNNCell的处理过程
with drnn.block():
    # 设置drnn的序列输入，并取得当前步的输入
    cur_input = drnn.step_input(input_data)

    # 设置memory变量，并取得上一时刻（或初始）隐状态
    last_hidden_state = drnn.memory(shape=[hidden_size], value=0.0)

    # 计算当前时刻隐状态
    cur_hidden_state = fluid.layers.fc(input=[cur_input, last_hidden_state], size=hidden_size, act='relu')

    # 更新隐状态
    drnn.update_memory(last_hidden_state, cur_hidden_state)

    # 记录本时刻的输出（BasicRNNCell中当前时刻的输出与当前时刻隐状态一致）
    drnn.output(hidden)

# 获取输出LoDTensor，其shape为(-1, hidden_size)
outputs = drnn()

# 获取各序列最后时刻的隐状态，其shape为（batch_size, hidden_size）
state = fluid.layers.sequence_last_step(outputs)
```

#### 其他相关op

&#160; &#160; &#160; &#160;为了简化用户定义动态RNN的过程，paddle有如下op可供选择：
- [paddle.fluid.layers.dynamic_lstm](http://www.paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#dynamic-lstm)：相当于tf.nn.dynamic_rnn结合tf.nn.rnn_cell.LSTMCell
- [paddle.fluid.layers.dynamic_gru](http://www.paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#dynamic-gru)：相当于tf.nn.dynamic_rnn结合tf.nn.rnn_cell.GRUCell
