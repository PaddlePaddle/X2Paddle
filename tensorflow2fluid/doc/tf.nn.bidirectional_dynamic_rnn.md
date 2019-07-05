## tf.nn.bidirectional_dynamic_rnn


### [tf.nn.bidirectional_dynamic_rnn](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/nn/bidirectional_dynamic_rnn)

```python
tf.nn.bidirectional_dynamic_rnn(
    cell_fw,
    cell_bw,
    inputs,
    sequence_length=None,
    initial_state_fw=None,
    initial_state_bw=None,
    dtype=None,
    parallel_iterations=None,
    swap_memory=False,
    time_major=False,
    scope=None
)
```

### 功能差异

#### 使用方式
TensorFlow：用户通过定义正向与反向`cell`，可以实现一个双向RNN网络的功能;  

PaddlePaddle：并没有提供一个对应的接口，用户可以使用`DynamicRNN`组合实现得到，详见如下代码示例。

### 代码示例
```
# 如下代码片段实现双向lstm网络，lstm单元数为16

num_unit_0 = 16

# 定义LoD输入
data = fluid.layers.data(name='input', shape=[1], dtype='int64', lod_level=1)

# 获得正向与反向embedding
embedding = fluid.layers.embedding(input=data, size=[emb_vocab, emb_size],
                                    is_sparse=False)
rev_embedding = fluid.layers.sequence_reverse(embedding)

# 定义lstm网络
def rnn(in_tensor):
  drnn = fluid.layers.DynamicRNN()
  with drnn.block():
          word = drnn.step_input(in_tensor) 
  
          prev_hid0 = drnn.memory(shape=[num_unit_0])
          prev_cell0 = drnn.memory(shape=[num_unit_0])
  
          cur_hid0, cur_cell0 = layers.lstm_unit(word, prev_hid0, prev_cell0)
  
          drnn.update_memory(prev_hid0, cur_hid0)  
          drnn.update_memory(prev_cell0, cur_cell0)  
          
          drnn.output(cur_hid0)

  out = drnn()
  return out

# 计算正向lstm网络的输出
out = rnn(embedding) 

# 计算反向lstm网络的输出
rev_out = rnn(rev_embedding) 

# 再次反转使得rev_out每个时刻所处理的数据与out对应
rev_rev_out = fluid.layers.sequence_reverse(rev_out)

# 合并得到最后的输出，其shape为(-1, 32)
concat_out = layers.concat([out, rev_rev_out], axis=1)                                               

```