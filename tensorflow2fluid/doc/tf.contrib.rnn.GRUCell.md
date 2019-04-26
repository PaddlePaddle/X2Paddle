## tf.contrib.rnn.GRUCell

### [tf.contrib.rnn.GRUCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/GRUCell)

```python
tf.contrib.rnn.GRUCell(
    num_units,
    activation=None,
    reuse=None,
    kernel_initializer=None,
    bias_initializer=None,
    name=None,
    dtype=None,
    **kwargs
)

```

### [paddle.fluid.layers.gru_unit](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#gru-unit)

```python
paddle.fluid.layers.gru_unit(
    input, 
    hidden, 
    size, 
    param_attr=None, 
    bias_attr=None, 
    activation='tanh', 
    gate_activation='sigmoid', 
    origin_mode=False
)
```

### 功能差异

#### 实现方式
TensorFlow：GRU的实现方式见论文[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/abs/1406.1078)；  
PaddlePaddle：GRU有两种实现方式，当设置`origin_mode=False`时，与TensorFlow实现方式一致；当设置`origin_mode=True`时，实现方式则参考论文[Empirical Evaluation of
Gated Recurrent Neural Networks
on Sequence Modeling](https://arxiv.org/pdf/1412.3555.pdf)。


#### 使用方式
TensorFlow：首先定义`GRUCell`对象，定义对象时只需要指定单元数`num_units`；由于`GRUCell`内部定义了`__call__`方法，因而其对象是可调用对象，直接使用`step_output, cur_state = cell(step_input, last_state)`的形式，可以计算得到当前步的输出与状态;  

PaddlePaddle：提供op形式的调用接口，通常与[paddle.fluid.layers.DynamicRNN](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#dynamicrnn)配合使用，以获取序列中的单步输入。**注意，为了提高`gru_unit`的计算效率，用户在使用该接口时需要遵从如下约定：假设要指定的GRU单元数为`num_units`，则`size`以及`input.shape[-1]`必须为`3*num_units`，`hidden.shape[-1]`为`num_units`，见如下代码示例小节。**

#### 返回值
TensorFlow：返回一个二元组，分别是当前时刻的输出值与隐藏状态，实际上输出值与隐藏状态为相同的tensor；  
PaddlePaddle：返回一个三元组，即`(hidden_value, reset_hidden_value, gate_value)`。后面两个元素为内部使用，用户可以只关注第一个元素。


### 代码示例
```
emb_size = 32                                                                                                                                                                                                                                 
emb_vocab = 10000                                                                                                                                                                                                                             
num_unit_0 = 10                                                                                                                                                                                                                               
                                                                                                                                                                                                                                              
data = fluid.layers.data(name='input', shape=[1], dtype='int64', lod_level=1)                                                                                                                                                                 
embedding = fluid.layers.embedding(input=data, size=[emb_vocab, emb_size],                                                                                                                                                                    
                                    is_sparse=False)                                                                                                                                                                                          

# 为了调用gru_unit，输入最后的维度必须为实际单元数的3倍
emb_fc = layers.fc(embedding, num_unit_0 * 3)                                                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                              
drnn = fluid.layers.DynamicRNN()                                                                                                                                                                                                              
with drnn.block():                                                                                                                                                                                                                            
    word = drnn.step_input(emb_fc) 
        
    # 指定上一时刻的隐状态，单元数为num_unit_0                                                                                                                                                                                                       
    prev_hid0 = drnn.memory(shape=[num_unit_0])                                                                                                                                                                                           
        
    # 执行gru_unit计算，num_unit_0 为实际的单元数                                                                                                                                                                                                                                      
    cur_hid0, _, _ = layers.gru_unit(word, prev_hid0, num_unit_0 * 3)
        
    # 更新隐状态                                                                                                                                                                                                                                              
    drnn.update_memory(prev_hid0, cur_hid0)                                                                                                                                                                                               
                                                                                                                                                                                                                                              
    drnn.output(cur_hid0)                                                                                                                                                                                                                 
                                                                                                                                                                                                                                              
out = drnn()                                                                                                                                                                                                                                  
last = fluid.layers.sequence_last_step(out)                       

```
