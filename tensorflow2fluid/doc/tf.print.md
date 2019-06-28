## tf.print

### [tf.print](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/print)

```python
tf.print(
    *inputs,
    **kwargs
)
```

### [paddle.fluid.layers.Print](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#print)
```python
paddle.fluid.layers.Print(
    input, 
    first_n=-1, 
    message=None, 
    summarize=-1, 
    print_tensor_name=True, 
    print_tensor_type=True, 
    print_tensor_shape=True, 
    print_tensor_lod=True, 
    print_phase='both'
)
```

### 功能差异

#### 使用方式
TensorFlow：在`graph`模式下，该op的运行决定于是否直接被运行，或者作为直接运行的其他op的依赖；在`eager`模式下，该op在被调用后会自动运行；  

PaddlePaddle：在被调用后，该op被添加到代码块，之后执行到代码块时将自动运行。

#### input类型
TensorFlow：可以是python primitives，也可以是tensor或其与python primitives的组合；  

PaddlePaddle：只可以是tensor。

#### 梯度打印
TensorFlow：不支持;  
PaddlePaddle：通过设置`print_phase`，可以控制是否打印`input`的梯度。


### 代码示例
```
# input 是任意paddle tensor

# 打印input的内容，如果有梯度的话也将打印梯度
print(input, message="content of input")
```