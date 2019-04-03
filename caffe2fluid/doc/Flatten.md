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


### [paddle.fluid.layers.reshape](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-72-reshape)
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
#### 输入参数
Caffe：分别使用参数`axis`和`end_axis`表示起始轴和结束轴，[axis, end_axis]轴上的数据将被压缩至一维，
但如若`axis-end_axis==1`时，则会在`axis`轴之后插入一维；
> 输入数据shape[2, 3, 4, 5]  
> axis=1, end_axis=3：输出shape[2, 60]  
> axis=3, end_axis=2：输出shape[2, 3, 4, 1, 5]  

PaddlePaddle：通过在`shape`参数设置具体的输出shape。
