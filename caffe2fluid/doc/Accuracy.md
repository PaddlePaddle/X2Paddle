## Accuracy


### [Accuracy](http://caffe.berkeleyvision.org/tutorial/layers/accuracy.html)
```
layer {
    name: "accuracy"
    type: "Accuracy"
    bottom: "pred"
    bottom: "label"
    top: "accuracy"
    include{
	phase: TEST
    }
}
```


### [paddle.fluid.layers.accuracy](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-253-accuracy)
```python
paddle.fluid.layers.accuracy(
    input,
    label,
    k = 1,
    correct = None,
    total = None
)
```  

### 功能差异
#### 计算机制的差异
Caffe：只能计算每个类别中top1中正确预测的个数。          
PaddlePaddle：可以通过设置`k`来计算每个类别中top k 中正确预测的个数。






