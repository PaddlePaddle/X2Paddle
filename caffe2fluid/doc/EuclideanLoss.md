## EuclideanLoss


### [EuclideanLoss](http://caffe.berkeleyvision.org/tutorial/layers/euclideanloss.html)
```
layer {
    name: "loss"
    type: "EuclideanLoss"
    bottom: "pred"
    bottom: "label"
    top: "loss"
}
```


### [paddle.fluid.layers.square_error_cost](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-167-square_error_cost)
```python
paddle.fluid.layers.square_error_cost(
    input,
    label
)
```  

### 功能差异
#### 实现方式
Caffe：对整个输入的欧氏距离进行取和后除以两倍的样本个数，最终获得一个标量数值。                                        

PaddlePaddle：使用elemenwise方式，计算`input`和`label`对应元素的欧式距离，最终获得一个array（输入和输出`shape`一致）： 

### 代码示例
```python
# 利用PaddlePaddle实现Caffe的EuclideanLoss
def EuclideanLoss(inputs, label):
    elw_eud = fluid.layers.square_error_cost(data, label)
    eud = fluid.layers.reduce_mean(elw_eud)
    eud = fluid.layers.scale(eud, scale=0.5)
    return eud

# 调用函数计算欧氏路离
# inputs: [1, 2, 4, 5, 6]
# labels: [6, 5, 4, 3, 2]
# eud: 5.4
inputs = fluid.layers.data(dtype='float32', shape=[5], name='data')
labels = fluid.layers.data(dtype='float32', shape=[5], name='label')
eud = EulideanLoss(inputs, labels)
```

