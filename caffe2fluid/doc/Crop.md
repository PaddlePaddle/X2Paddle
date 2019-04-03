## Crop


### [Crop](http://caffe.berkeleyvision.org/tutorial/layers/crop.html)
```
layer {
    name: "crop"
    type: "Crop"
    bottom: "data1"
    bottom: "data2"
    top: “crop"
    crop_param{
        axis: 1
        offset: 0
        offset: 2
    }
}
```


### [paddle.fluid.layers.crop](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-51-crop)
```python
paddle.fluid.layers.crop(
    x, 
    shape=None, 
    offsets=None, 
    name=None
)
```  

### 功能差异
#### 输出大小
Caffe：输入为`data1`，裁剪的输出大小与`data2`(Variable类型)一致；              
PaddlePaddle：`shape`参数支持python list的方式传入输出大小，同时也支持`Variable`类型的输入。当`shape`为`Variable`类型时，用法与Caffe类似，裁剪输出大小与`shape`参数的大小一致。

#### 裁剪偏移量
Caffe：只需要设置需要裁剪的维度的偏移量。             
PaddlePaddle：每一个维度需要设置偏移量。
### 代码示例
```  
# Caffe示例： 
# data1 shape：(20，3，128，128)
# data2 shape：(20，2，64，64)
layer {
    name: "crop"
    type: "Crop"
    bottom: "data1"
    bottom: "data2"
    top: ”crop"
    crop_param{
        axis: 1
        offset: 0
        offset: 25
        offset: 25
    }
}
# 输出shape：(20，2，64，64)
```  
```python
# PaddlePaddle示例：  
# inputs1输入shape：(20，3，128，128)
output1 = fluid.layers.crop(x = inputs1, shape=inputs2, offsets=[0,0,25,25])
# 输出shape：(20，2，64，64)
output = fluid.layers.crop(x = inputs1, shape=[20,2,64,64], offsets=[0,0,25,25])
```
