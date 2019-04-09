## Sofmax


### [Softmax](http://caffe.berkeleyvision.org/tutorial/layers/softmax.html)
```
layer {
    name: "softmax"
    type: "Softmax"
    bottom: "fc"
    top: "softmax"	
}
```


### [paddle.fluid.layers.softmax](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-163-softmax)
```python
paddle.fluid.layers.softmax(
    input, 
    use_cudnn=True, 
    name=None
)
```  

### 功能差异
#### 计算机制
Caffe：计算softmax之前，对每个样本中的每个值减去该样本中的最大值;                 
PaddlePaddle：省略了这一操作直接计算softmax。
