## Reduction


### [Reduction](http://caffe.berkeleyvision.org/tutorial/layers/reshape.html)
```
layer {
    name: "reduce"
    type: "Reduction"
    bottom: "reduce"
    top: “reduce"
    reduction_param{
        operation: SUM
	axis: 1
	coeff: 2
    }
}
```


### [paddle.fluid.layers.reduce_sum](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-127-reduce_sum)、[paddle.fluid.layers.reduce_mean](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-124-reduce_mean)
```python
paddle.fluid.layers.reduce_sum(
    input, 
    dim=None, 
    keep_dim=False, 
    name=None
)
和
paddle.fluid.layers.reduce_mean(
    input, 
    dim=None, 
    keep_dim=False, 
    name=None
)
```  

### 功能差异
#### 输入参数的差异
Caffe：一个层里面可以是`SUM`、`ASUM`、`SUMSQ`或者`MEAN`这四种操作。                                          
PaddlePaddle：只能完成里面的两种操作。同时Caffe可以设置`coeff`来将每个值乘以一个系数。

#### 输出的差异
Caffe：`axis`往后的每个维度都会缩减为一个维度。              
PaddlePaddle：只会缩减`dim`中list定义的维度，并根据`keep_dim`确定是否在输出Tensor中保留减小的维度。
### 代码示例
```  
# Caffe示例：  
# 输入shape：(30，3，6，8)
layer {
    name: "reduce"
    type: "Reduction"
    bottom: "reduce"
    top: “reduce"
    reduction_param{
	operation: SUM
	axis: 2
	coeff: 2
    }
}
# 输出shape：(30,3,)
```  
```python 
# PaddlePaddle示例：  
# 输入shape：(30，3，6，8)
output1 = fluid.layers.reduce_mean(input = inputs, dim=[1])
# 输出shape：(30,6,8)
output2 = fluid.layers.reduce_mean(input = inputs, dim=[1], keep_dim=True, name=None)
# 输出shape：(30,1,6,8)
```  
