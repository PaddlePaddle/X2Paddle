## Reshape


### [Reshape](http://caffe.berkeleyvision.org/tutorial/layers/reshape.html)
```
layer {
    name: "reshape"
    type: "Reshape"
    bottom: "data"
    top: "reshape"
    reshape_param {
	shape{
	    dim: 1
	    ...
	}
	axis: 0
	num_axes: -1
    }
}
```


### [paddle.fluid.layers.reshape](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-130-reshape)
```python
paddle.fluid.layers.reshape(
    x, 
    shape, 
    actual_shape=None, 
    act=None, 
    inplace=False, 
    name=None
)
```  

### 功能差异
#### reshape机制的差异
Caffe：使用0和-1分别代表复制的维度数和推断的维度数，但使用了`axis`和`num_axes`定义了其他的使用方法。当单独使用`axis`时，表示输出数据的前`axis`个维度由原始输入数据的前`axis`个维度复制而来，而`shape`里的维度信息则添加在这几个维度之后；当同时使用`axis`和`num_axes`两个参数时，表示`shape`中的第`1`个维度至第`1+num_axes`维度定义为输出中的第`axis+1`和`axis+num_axes+1`个维度，其余维度的维度数由原始输入数据的维度数代替，直至输出数据和输入数据摊平成一维时大小相同。   
PaddlePaddle：使用0和1分别代表复制的维度数和推断的维度数。


#### 输出的差异
Caffe：Reshape层在不改变数据的情况下改变输入blob的维度，处理过程只在输入blob上进行，没有进行数据的拷贝。            
PaddlePaddle：可以通过设置`inplace`表示是否对数据进行拷贝。
#### 其他差异
Caffe：激活函数需要由另外一层完成。            
PaddlePaddle：可以通过设置`act`对reshpe后的tensor变量执行非线性激活。



### 代码示例
```  
# Caffe示例：  
# 输入shape：(2,4,6)
layer {
    name: "reshape"
    type: "Reshape"
    bottom: "data"
    top: "reshape"
    reshape_param {
	shape {
	    dim: 3
	    dim: 2
	}
	axis: 2
	num_axes: 1
    }
}
# 输出shape：(2,4,3,2）
layer {
    name: "reshape"
    type: "Reshape"
    bottom: "data"
    top: "reshape"
    reshape_param {
	shape {
	    dim: 3
	    dim: 2
	    dim: 4
	}
	axis: 1
    }
}
# 输出shape：(2,3,2,4)

```  
```python 
# PaddlePaddle示例：  
# 输入shape：(2,4,6)
output1 = paddle.fluid.layers.reshape(x = inputs , shape = [2,4,-1,3])
# 输出shape：(2,4,2,3)
output2 = paddle.fluid.layers.reshape(x = inputs , axis = [0,2,2,6])
# 输出shape：(2,2,2,6)
```  
