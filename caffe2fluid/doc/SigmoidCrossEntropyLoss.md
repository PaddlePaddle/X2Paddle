## SigmoidCrossEntropyLoss


### [SigmoidCrossEntropyLoss](http://caffe.berkeleyvision.org/tutorial/layers/sigmoidcrossentropyloss.html)
```
layer {
    name: "loss"
    type: "SigmoidCrossEntropyLoss"
    bottom: "pred"
    bottom: "label"
    top: "loss"
}
```


### [paddle.fluid.layers.sigmoid_cross_entropy_with_logits](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-158-sigmoid_cross_entropy_with_logits)
```python
paddle.fluid.layers.sigmoid_cross_entropy_with_logits(
    x, 
    label, 
    ignore_index=-100, 
    name=None, 
    normalize=False
)
```  

### 功能差异
#### 输入格式
Caffe：输入的数据维度最大是4维`N*C*H*W`；                 
PaddlePaddle：输入只能是2维`N*H`。

#### 输出格式
Caffe：输出的数据大小是`1*1*1*1`，即将所有位置上的loss取均值；                      
PaddlePaddle：输出和输入大小一致，即`N*H`。
#### 其他
PaddlePaddle：可以通过设定`ignore_index`来确定忽略的目标值，同时它有一个`normalize`参数可以输出除以除去`ignore_index`对应目标外的目标数所得的结果。

