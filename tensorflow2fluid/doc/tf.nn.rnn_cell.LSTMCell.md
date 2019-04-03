## tf.nn.rnn_cell.LSTMCell

### [tf.nn.rnn_cell.LSTMCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/LSTMCell)

```python
tf.nn.rnn_cell.LSTMCell(
    num_units,
    use_peepholes=False,
    cell_clip=None,
    initializer=None,
    num_proj=None,
    proj_clip=None,
    num_unit_shards=None,
    num_proj_shards=None,
    forget_bias=1.0,
    state_is_tuple=True,
    activation=None,
    reuse=None,
    name=None,
    dtype=None,
    **kwargs
)
```

### [paddle.fluid.layers.lstm_unit](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#lstm-unit)

```python
paddle.fluid.layers.lstm_unit(
    x_t, 
    hidden_t_prev, 
    cell_t_prev, 
    forget_bias=0.0, 
    param_attr=None, 
    bias_attr=None, 
    name=None
)
```

### 功能差异

#### 使用方式
TensorFlow：首先定义`LSTMCell`对象，定义对象时只需要指定单元数`num_units`；由于`LSTMCell`内部定义了`__call__`方法，因而其对象是可调用对象，直接使用`step_output, cur_state = cell(step_input, last_state)`的形式，可以计算得到当前步的输出与状态;  

PaddlePaddle：提供op形式的调用接口，通常与[paddle.fluid.layers.DynamicRNN](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#dynamicrnn)配合使用，以获取序列中的单步输入。**注意，`lstm_unit`通过`cell_t_prev`最后一个维度来确定lstm的单元数，同时要求`hidden_t_prev`与`cell_t_prev`最后的维度相同。**

#### 窥孔连接

TensorFlow：通过设置`use_peepholes`选择LSTM的实现是否进行窥孔连接;  
PaddlePaddle：只提供非窥孔连接的LSTM实现。

#### 输出变换
TensorFlow：第一个返回值为`step_output`。当`num_proj`非空时，由`hidden_state`经过`fc`变换后得到`step_output`；而当`num_proj`为空时，则直接返回`hidden_step`作为`step_output`；   
PaddlePaddle：第一个返回值为`hidden_state`，不涉及输出变换。

#### cell_state
TensorFlow：第二个返回值为`cell_state`。`cell_state`由真实的`cell_state`与`hidden_state`一起构成：当`state_id_tuple`为`True`时，返回真实的`cell_state`与`hidden_state`组成的`tuple`；反之，则返回`concat([cell_state, hidden_state], axis=1)`；
PaddlePaddle：第二个返回值为真实的`cell_state`。

### 代码示例
```
# embedding 是一个rank为2，lod_level为1的LoDTensor

num_unit_0 = 32
drnn = fluid.layers.DynamicRNN()                                                                                                                                                                                                              
with drnn.block():                                                                                                                                                                                                                            
        word = drnn.step_input(embedding)       
        
        # 记录hidden_state与cell_state，初始状态使用零向量                                                                                                                                                                                              
        prev_hid0 = drnn.memory(shape=[num_unit_0])                                                                                                                                                                                           
        prev_cell0 = drnn.memory(shape=[num_unit_0])                                                                                                                                                                                          
        
        # 执行lstm计算                                                                                                                                                                                                                                      
        cur_hid0, cur_cell0 = layers.lstm_unit(word, prev_hid0, prev_cell0)                                                                                                                                                                   
        
        # 更新hidden_state与cell_state                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        drnn.update_memory(prev_hid0, cur_hid0)                                                                                                                                                                                               
        drnn.update_memory(prev_cell0, cur_cell0)                                                                                                                                                                                             
        
        # 输出每个时刻的hidden_state                                                                                                                                                                                                                                      
        drnn.output(cur_hid0)

# 获取每个时刻的输出
out = drnn()

# 获取最后时刻的输出
last = fluid.layers.sequence_last(out)                                                  

```
