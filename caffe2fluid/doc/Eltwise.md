## Eltwise


### [Eltwise](http://caffe.berkeleyvision.org/tutorial/layers/eltwise.html)
```
layer {
    name: "eltwise"
    type: "Eltwise"
    bottom: "num1"
    bottom: "num2"
    top: "prod"
    eltwise_param {
        operation: PROD    # 还有MAX，SUM
        stable_prod_grad: false
        # coeff: 1
        # coeff: -1
    }
}
```


### [paddle.fluid.layers.elementwise_sum](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-61-elementwise_add)
### [paddle.fluid.layers.elementwise_max](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-63-elementwise_max)
### [paddle.fluid.layers.elementwise_mul](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-65-elementwise_mul)
```python
paddle.fluid.layers.elementwise_sum(
    x, 
    y, 
    axis=-1, 
    act=None,
    name=None
)
和
paddle.fluid.layers.elementwise_max(
    x, 
    y, 
    axis=-1, 
    act=None,
    name=None
)
和
paddle.fluid.layers.elementwise_mul(
    x, 
    y, 
    axis=-1, 
    act=None,
    name=None
)
```  

### 功能差异
#### 输入数据
Caffe：`num1`和`num2`的`shape`必须按相同；          
PaddlePaddle：`Y`的`shape`可以是`X`的`shape`可以的一个连续子序列，并通过设置`axis`表示从哪一个维度开始对应。

#### 加法操作的差异
Caffe：可以通过设置`coeff`参数为加法的每个输入添加一个权重；       
PaddlePaddle：无权重设置功能。

#### 乘法操作
Caffe：可以通过设置`stable_prod_grad`参数来选择是否渐进较慢的梯度计算方法；                     
PaddlePaddle：无设置`stable_prod_grad`参数的功能。

#### 其他
Caffe：激活函数需要由另外一层完成；               
PaddlePaddle：可以通过设置`act`对逐元素操作后的tensor变量执行非线性激活。
