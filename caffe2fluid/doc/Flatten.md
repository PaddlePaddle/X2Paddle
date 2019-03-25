## Flatten


### [Flatten](http://caffe.berkeleyvision.org/tutorial/layers/flatten.html)
```
layer {
	name: "flatten"
	type: "Flatten"
	bottom: "data"
	top: "flatten"
	flatten_param{
		axis: 1
		end_axis: -1
	}
}
```


### [paddle.fluid.layers.flatten](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-72-flatten)
```python
paddle.fluid.layers.flatten(
	x,
	axis = 1,
	name = None
)
```  

### 功能差异
#### 转换机制的差异
Caffe：有两个参数，`axis`代表转换起始点，`end_axis`代表转换终止点，假设输入数据的维度为n，则`axis`和`end_axis`的取值范围都是[-n,n-1]（其中当i是一个大于等于-n的负值时，可以将其等同于i+n）。它有两种用法：当`axis<=end_axis`时，代表将第`axis+1`维数据至第`end_axis+1`维数据压缩至同一纬度的数据；当`axis`是一个大于等于-n的负值或者0且`end_axis=axis+n-1`时，代表在第`end_axis+1`个维度插入一个维度，且该维度大小为1，其余维度后移。  
PaddlePaddle：只有一个参数`axis`,其取值范围为[0,n]，小于等于`axis`的维度压缩成一个维度，剩下的压缩成另一个维度，当某一边维度数为0时，则添入一个维度大小为1的维度。  
### 代码示例
```  
# Caffe代码示例：
# 输入shape：(10,3,5,5)  
layer {
	name: "flatten"
	type: "Flatten"
	bottom: "data"
	top: "flatten"
	flatten_param{
		axis: 1
		end_axis: -2
	}
}
# 输出shape：(10,15,10）
layer {
	name: "flatten"
	type: "Flatten"
	bottom: "data"
	top: "flatten"
	flatten_param{
		axis: 1
		end_axis: -2
	}
}
# 输出shape：(10,3,5,1,5）

```  
```python
# PaddlePaddle示例：  
# 输入shape：(10,3,5,5)  
output1 = paddle.fluid.layers.flatten(x = inputs , axis = 2)
# 输出shape：(30,15)
output2 = paddle.fluid.layers.flatten(x = inputs , axis = 4)
# 输出shape：(450,1)
```  
