## Eltwise


### [Eltwise](http://caffe.berkeleyvision.org/tutorial/layers/eltwise.html)
```
layer {
    name: "eltwise"
    type: "Eltwise"
    bottom: "num1"
    bottom: "num2"
    top: "prod"
    eltwise_param{
        operation: PROD	#还有MAX，SUM
        stable_prod_grad: false
        # coeff: 1
        # coeff: -1
    }
}
```


### [paddle.fluid.layers.elementwise_sum](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-61-elementwise_add)[paddle.fluid.layers.elementwise_max](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-63-elementwise_max)[paddle.fluid.layers.elementwise_mul](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-65-elementwise_mul)
```python
paddle.fluid.layers.elementwise_sum(
    x, 
    y, 
    axis = -1, 
    act = None,
    name = None
)
和
paddle.fluid.layers.elementwise_max(
    x, 
    y, 
    axis = -1, 
    act = None,
    name = None
)
和
paddle.fluid.layers.elementwise_mul(
    x, 
    y, 
    axis = -1, 
    act = None,name = None
)
```  

### 功能差异
#### 输入数据的差异
假设逐元素操作的两个输入分别是`X`和`Y`。
Caffe：`X`和`Y`的`shape`必须按相同，否则会出错。                
PaddlePaddle：`Y`的`shape`可以是`X`的`shape`可以的一个连续子序列，并通过设置`axis`表示从哪一个维度开始对应。

#### 加法操作的差异
Caffe：可以通过设置`coeff`参数为加法的每个输入添加一个权重，所以实际上其可以完成剪发操作。              
PaddlePaddle：无权重设置功能。
#### 乘法操作的差异
Caffe：可以通过设置`stable_prod_grad`参数来选择是否渐进较慢的梯度计算方法。                   
PaddlePaddle：无设置`stable_prod_grad`参数的功能。
#### 其他差异
Caffe：激活函数需要由另外一层完成。            
PaddlePaddle：可以通过设置`act`对逐元素操作后的tensor变量执行非线性激活。

### 代码示例
``` 
# Caffe示例：  
# 输入num1的shape：(2,3,4,5)
# 输入num2的shape：(2,3,4,5)
layer {
	name: "eltwise"
	type: "Eltwise"
	bottom: "num1"
	bottom: "num2"
	top: "sum"
	eltwise_param{
		operation: SUM
		coeff: 1
		coeff: 1
	}
}
# 输出shape：(2,3,4,5)
```  
```python
# PaddlePaddle示例：  
# 输入num1的shape：(2,3,4,5)
# 输入num2的shape：(3,4)
output = paddle.fluid.layers.elementwise_sum(x = num1, y = num2, axis = 1)
# 输出shape：(2,3,4,5)
```  
