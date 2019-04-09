## Dropout


### [Dropout](http://caffe.berkeleyvision.org/tutorial/layers/dropout.html)
```
layer {
    name: "dropout"
    type: "Dropout"
    bottom: "data"
    top: “dropout"
    dropout_param {
	dropout_ratio: 0.5
    }
}
```


### [paddle.fluid.layers.dropout](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-56-dropout)
```python
paddle.fluid.layers.dropout(
    x, 
    dropout_prob, 
    is_test=False, 
    seed=None, 
    name=None, 
    dropout_implementation="downgrade_in_infer"
)
```  

### 功能差异
#### 实现方式
Caffe：采用`upscale_in_train`方式实现；             
PaddlePaddle：实现方式支持`downgrade_in_infer`和`upscale_in_infer`两种方式。
```
1. downgrade_in_infer实现方式
    训练时： out = input * mask
    预测时： out = input * dropout_prob
2. upscale_in_infer实现方式
    训练时： out = input * mask / (1.0 - dropout_prob)
    预测时： out = input
```
