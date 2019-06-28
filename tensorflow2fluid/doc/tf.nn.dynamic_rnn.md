## tf.nn.dynamic_rnn

### [tf.nn.dynamic_rnn](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/nn/dynamic_rnn)
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

### [paddle.fluid.layers.DynamicRNN](http://www.paddlepaddle.org/documentation/docs/zh/1.4/api_cn/api_guides/low_level/layers/control_flow.html#dynamicrnn)
``` python
paddle.fluid.layers.DynamicRNN(name=None)
```

### 功能差异
#### 调用机制
Tensorflow: `tf.nn.dynamic_rnn`通常与`tf.nn.rnn_cell.LSTMCell`、`tf.nn.rnn_cell.GRUCell`等Cell结合使用  
PaddlePaddle: 使用`paddle.fluid.layers.DynamicRNN`类实现类似功能 ，通过DynamicRNN提供的类方法，用户可以在`with block`中方便地自定义每个时间步的处理过程。

#### 输入格式
TensorFlow: `tf.nn.dynamic_rnn`输入为序列数据，批输入中的每个序列需要填充到相同的长度
PaddlePaddle: 使用
[LoDTensor](http://www.paddlepaddle.org/documentation/docs/zh/1.4/user_guides/howto/basic_concept/lod_tensor.html)表示一个批输入，用户在使用时不需要进行填充操作。

### 代码示例

```
# TensorFlow代码示例
# 创建 BasicRNNCell
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
# 定义初始隐状态
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
# 输出shape为（batch_size, max_time, cell_state_size）
# 最后时刻隐状态shape为（batch_size, cell_state_size）
outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                   initial_state=initial_state,
                                   dtype=tf.float32)

# PaddlePaddle代码示例
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

### 其他

为了简化用户定义动态RNN的过程，paddle有如下op可供选择：
- [paddle.fluid.layers.dynamic_lstm](http://www.paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#dynamic-lstm)：相当于  `tf.nn.dynamic_rnn`结合`tf.nn.rnn_cell.LSTMCell`
- [paddle.fluid.layers.dynamic_gru](http://www.paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#dynamic-gru)：相当于`tf.nn.dynamic_rnn`结合`tf.nn.rnn_cell.GRUCell`