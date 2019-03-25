## Input
### [Input](http://caffe.berkeleyvision.org/tutorial/layers/input.html)
```
layer {
    name: "input"
    type: "Input"
    top: "input"	
    input_param{
        shape{
	    dim: 10
	    dim: 3
	    dim: 227
	    dim: 227
	}
    }
}
```


### [paddle.fluid.layers.data](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-20-data)
```python
paddle.fluid.layers.data(
    name, 
    shape, 
    append_batch_size=True, 
    dtype='float32', 
    lod_level=0, 
    type=VarType.LOD_TENSOR, 
    stop_gradient=True
)
```  

### 功能差异
#### 输入shape的差异
Caffe：输入的shape中每一个维度的大小都需要详细定义。  
PaddlePaddle：可以根据设置设置`append_batch_size`来确定是否将数据第一个维度的大小加入到shape中，若该参数为True，输入数据第一个维度的大小则由传入数据决定，若该参数为False，则shape的第一个维度为输入数据第一个维度的大小。   



#### 其他差异
Caffe：不需要强制定义输入数据的类型。  
PaddlePaddle：需要强制定义输入数据的类型，同时可以通过设置`lod_level`表示输入的数据是不是一个序列，设置`stop_gradient`表示是否应该停止计算梯度。


### 代码示例
``` python
# Caffe示例：
layer{
    name: "input"
    type: "Input"
    top: "input"	
    input_param{
    	shape{
	    dim: 10
	    dim: 3
	    dim: 227
	    dim: 227
	}
    }
}
# 数据shape为[10,3,227,227]


# PaddlePaddle示例：
inputs1 = paddle.fluid.layers.data(name = 'data1', shape = [10,3,227,227], dtype = 'float32', append_batch_size = False)
# 数据shape为[10,3,227,227]
inputs2 = paddle.fluid.layers.data(name = 'data2', shape = [3,227,227], dtype = 'float32')
# 数据shape为[-1,3,227,227]
```  
