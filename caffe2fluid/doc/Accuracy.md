## Accuracy


### [Accuracy](http://caffe.berkeleyvision.org/tutorial/layers/accuracy.html)
```
layer {
    name: "accuracy"
    type: "Accuracy"
    bottom: "input"
    bottom: "label"
    top: "accuracy"
    accuracy_param{
        top_k = 1
	axis = 1
	# ignore_label = 0 # 定义需要忽略的label的类标，一般未定义
    }
}
```


### [paddle.fluid.layers.accuracy](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#accuracy)
```python
paddle.fluid.layers.accuracy(
    input,
    label,
    k=1,
    correct=None,
    total=None
)
```  

### 功能差异
#### 计算机制
Caffe：可以设置计算某一维的accuracy；          
PaddlePaddle：只能计算axis为1的维度的accuracy。






