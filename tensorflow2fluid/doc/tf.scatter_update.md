### tf.scatter_update

#### [tf.scatter_update](https://www.tensorflow.org/api_docs/python/tf/scatter_update)

```python
tf.scatter_update(
    ref,
    indices,
    updates,
    use_locking=True,
    name=None
)
```

#### [paddle.fluid.layers.scatter](http://paddlepaddle.org/documentation/docs/zh/1.2/api_cn/layers_cn.html#scatter)

```python
paddle.fluid.layers.scatter(
    input, 
    index, 
    updates, 
    name=None
)
```

#### 功能差异

##### 参数类型

Tensorflow：`indices`支持任意维度，可以是变量，也可以是常量;  
PaddlePaddle：`index`只支持1-d Variable。

##### 其他
Tensorflow：`updates`支持numpy-style broadcasting;  
PaddlePaddle：`updates`要求其rank与`input`相同，同时`updates.shape[0]`等于`index.shape[0]`。

#### paddlepaddle代码示例
```
# x是dtype为float32, shape为[3,9,5]的张量    

# 将x[1:,:,:]置为1，并返回更新后的张量
out = layers.scatter(x, 
  index=layers.assign(np.array([1,2], dtype='int32')),                                                                                                                                                          
  updates=layers.assign(np.ones((2,9,5), dtype='float32')))        

```
