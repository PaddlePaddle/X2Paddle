## Reduction


### [Reduction](http://caffe.berkeleyvision.org/tutorial/layers/reshape.html)
```
layer {
    name: "reduce"
    type: "Reduction"
    bottom: "reduce"
    top: “reduce"
    reduction_param {
        operation: SUM
	axis: 1
	coeff: 2
    }
}
```


### [paddle.fluid.layers.reduce_sum](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-127-reduce_sum)
### [paddle.fluid.layers.reduce_mean](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-124-reduce_mean)
```python
paddle.fluid.layers.reduce_sum(
    input, 
    dim=None, 
    keep_dim=False, 
    name=None
)
```
```python
paddle.fluid.layers.reduce_mean(
    input, 
    dim=None, 
    keep_dim=False, 
    name=None
)
```  

### 功能差异
#### 操作类型
Caffe：通过`operation`参数支持`SUM`、`ASUM`、`SUMSQ`、`MEAN`四种操作；                                          
PaddlePaddle：`reduce_sum`和`reduce_mean`分别对应Caffe的`SUM`和`MEAN`操作，另外两种无对应。

#### 计算方式
Caffe：`axis`为`int`型参数，该维及其后维度，均会被降维，且不保留对应部分的维度，如shape为`(30， 3， 6， 8)`， `axis`为2的情况下，得到的输出shape为`(30, 3)`；              
PaddlePaddle：`dim`参数为`list`型参数，其指定的维度才会被降维，且当`keep_dim`为`True`时，降维的维度仍会以`1`的形式保留下来，如shape为`(30, 3, 6, 8)`， `dim`为`[2, 3]`，`keep_dim`为`True`的情况下，得到的输出shape为`(30, 3, 1, 1)`。

### 代码示例
```  
# Caffe示例：  
# 输入shape：(30，3，6，8)
layer {
    name: "reduce"
    type: "Reduction"
    bottom: "reduce"
    top: “reduce"
    reduction_param {
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
