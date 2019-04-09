## tf.nn.rnn_cell.MultiRNNCell

### [tf.nn.rnn_cell.MultiRNNCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/MultiRNNCell)

```python
tf.nn.rnn_cell.MultiRNNCell(
    cells,
    state_is_tuple=True
)
```

### PaddlePaddle实现
在Tensorflow中，用户通过定义多个单独的`RNNCell`生成一个`cell`列表，进而调用`MultiRNNCell`，可以实现一个多层RNN网络的功能。PaddlePaddle并没有提供一个对应的接口，用户可以在`DynamicRNN`的block中，通过组合多个RNN相关的`unit`实现类似的功能，可参考代码示例。


### 代码示例
```
# 如下代码片段实现两层lstm网络，第一层单元数为32，第二层单元数为16
num_unit_0 = 32
num_unit_1 = 16

emb_size = 12
emb_vocab = 10000

data = fluid.layers.data(name='input', shape=[1], dtype='int64', lod_level=1)
embedding = fluid.layers.embedding(input=data, size=[emb_vocab, emb_size])
                                    
drnn = fluid.layers.DynamicRNN()
with drnn.block():
    # 定义单步输入
    word = drnn.step_input(embedding)
        
    # 定义第一层lstm的hidden_state, cell_state
    prev_hid0 = drnn.memory(shape=[num_unit_0])
    prev_cell0 = drnn.memory(shape=[num_unit_0])
        
    # 定义第二层lstm的hidden_state, cell_state
    prev_hid1 = drnn.memory(shape=[num_unit_1])
    prev_cell1 = drnn.memory(shape=[num_unit_1])

    # 执行两层lstm运算
    cur_hid0, cur_cell0 = layers.lstm_unit(word, prev_hid0, prev_cell0)
    cur_hid1, cur_cell1 = layers.lstm_unit(cur_hid0, prev_hid1, prev_cell1)

    # 更新第一层lstm的hidden_state, cell_state
    drnn.update_memory(prev_hid0, cur_hid0)  
    drnn.update_memory(prev_cell0, cur_cell0)  
       
    # 更新第二层lstm的hidden_state, cell_state
    drnn.update_memory(prev_hid1, cur_hid1)  
    drnn.update_memory(prev_cell1, cur_cell1)  

    drnn.output(cur_hid1)

out = drnn()
last = fluid.layers.sequence_last_step(out)
                                               

```
