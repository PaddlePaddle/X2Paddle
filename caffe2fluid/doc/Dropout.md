## Dropout


### [Dropout](http://caffe.berkeleyvision.org/tutorial/layers/dropout.html)
```
layer {
    name: "dropout"
    type: "Dropout"
    bottom: "data"
    top: “dropout"
    dropout_param{
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
    dropout_implementation='downgrade_in_infer'
)
```  

### 功能差异
#### 输入参数的差异
Caffe：输出的是PaddlePaddle中`dropout_implementation`设置为`upscale_in_train`的结果。               
PaddlePaddle：相对于Caffe，多使用了`seed`、`dropout_implementation`和`is_test`几个参数。
