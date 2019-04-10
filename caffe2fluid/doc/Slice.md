## Slice


### [Slice](http://caffe.berkeleyvision.org/tutorial/layers/slice.html)
```
layer {
    name: "slice"
    type: "Slice"
    bottom: "data"
    top: "out1"
    top: "out2"
    top: "out3"
    slice_param {
	axis: 1
	alice_point: 1
	alice_point: 2
    }
}
```


### [paddle.fluid.layers.slice](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-160-slice)
```python
paddle.fluid.layers.slice(
    input, 
    axes, 
    starts, 
    ends
)
```  

### 功能差异
#### 输入参数
Caffe：输入的`axis`和`alice_point`等参数都是数值。               
PaddlePaddle：输入的`axes`、`starts`和`ends`等输入参数都是list类型。
#### slice机制
Caffe：只能在一个维度上截取，但可以截取多个切片。            
PaddlePaddle：可以在多个维度上截取，但只能截取到一个切片。
#### 其他差异
PaddlePaddle：如果传递给`starts`或`end`的值大于n（此维度中的元素数目），则表示n。
### 代码示例
```  
# Caffe示例：  
# 输入shape：(2,6)
layer {
    name: "slice"
    type: "Slice"
    bottom: "data"
    top: "out1"
    top: "out2"
    top: "out3"
    slice_param {
	axis: 1    # 使用-1效果相同
	alice_point: 1
	alice_point: 2
    }
}
# 输出3个数组，第一个shape：(2,1)，第二个shape：(2,1)，第三个shape：(2,4)
```  
```python
# PaddlePaddle示例：  
# 输入shape：(2,6)
output1 = paddle.fluid.layers.slice(input=inputs, axes=[1], starts=[1], ends=[3])
# 输出shape：(2，2)
output2 = paddle.fluid.layers.slice(input=inputs, axes=[0,1], starts=[0,1], ends=[1,3])
# 输出shape：(1,2)
```  
