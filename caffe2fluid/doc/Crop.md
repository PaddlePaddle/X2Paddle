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
#### 裁剪参考输入的差异
Caffe：裁剪参考输入只能是Variable的格式。              
PaddlePaddle：剪裁参考输入可以是Variable，也可以是一个list或者tuple，其中放入每一个维度的维度数。
#### 裁剪偏移量输入的差异
Caffe：只需要设置需要裁剪的维度的偏移量。             
PaddlePaddle：每一个维度需要设置偏移量。
### 代码示例
```  
# Caffe示例： 
# data1输入shape：(20，3，128，128)
# data2输入shape：(20，2，64，64)
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
# inputs2输入shape：(20，2，64，64)
output1 = fluid.layers.crop(x = inputs1, shape=inputs2, offsets=[0,0,25,25])
# 输出shape：(20，2，64，64)
output = fluid.layers.crop(x = inputs1, shape=[20,2,64,64], offsets=[0,0,25,25])
# 输出shape：(20，2，64，64)，其与output1输出结果一致
```
