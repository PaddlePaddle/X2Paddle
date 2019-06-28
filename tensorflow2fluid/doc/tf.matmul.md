## tf.matmul

### [tf.matmul](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/linalg/matmul)
``` python
tf.matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    name=None
)
```

### [paddle.fluid.layers.matmul](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#matmul)
``` python
paddle.fluid.layers.matmul(
    x, 
    y, 
    transpose_x=False, 
    transpose_y=False, 
    alpha=1.0, 
    name=None
)
```

### 功能差异
#### 输入格式
TensorFlow：要求op的两个操作数具有相同的rank；  
PaddlePaddle：允许两者具有不同的rank，具体说就是当任一操作数的rank大于2时，将其看做最里面两维度矩阵的堆叠，paddlepaddle将进行broadcast操作。

#### 其他
TensorFlow：使用`adjoint`参数可以实现快速的共轭操作；paddlepaddle中并不支持；  
PaddlePaddle：额外支持对输出进行数乘操作。


### 代码示例
```python
# x: [M, K], y: [K, N]
fluid.layers.matmul(x, y)  # out: [M, N]

# x: [B, ..., M, K], y: [B, ..., K, N]
fluid.layers.matmul(x, y)  # out: [B, ..., M, N]

# x: [B, M, K], y: [B, K, N]
fluid.layers.matmul(x, y)  # out: [B, M, N]

# x: [B, M, K], y: [K, N]
fluid.layers.matmul(x, y)  # out: [B, M, N]

# x: [B, M, K], y: [K]
fluid.layers.matmul(x, y)  # out: [B, M]
        
# x: [K], y: [K]
fluid.layers.matmul(x, y)  # out: [1]

# x: [M], y: [N]
fluid.layers.matmul(x, y, True, True)  # out: [M, N]
```