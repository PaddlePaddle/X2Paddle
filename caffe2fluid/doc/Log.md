## Log


### [Log](http://caffe.berkeleyvision.org/tutorial/layers/log.html)
```
layer {
    name: "log"
    type: "Log"
    bottom: "data"
    top: "log"
    log_param{
        base: -1
        scale: 1
	shift: 0
    }
}
```


### [paddle.fluid.layers.log](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-93-log)
```python
paddle.fluid.layers.log(
    x,
    name=None
)
```  

### 功能差异
#### 计算机制的差异

Caffe：有三个关于计算的参数，其计算公式为：  
$$
y=\begin{cases}
ln(shift+scale \times x),\quad base\leq 0 \\\\
log_base(shift+scale \times x),\quad base>0
\end{cases}
$$              
             
PaddlePaddle：计算公式为：$$y=ln(x)$$
