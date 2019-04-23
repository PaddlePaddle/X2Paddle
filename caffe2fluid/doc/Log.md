## Log


### [Log](http://caffe.berkeleyvision.org/tutorial/layers/log.html)
```
layer {
    name: "log"
    type: "Log"
    bottom: "data"
    top: "log"
    log_param {
        base: -1
        scale: 1
	shift: 0
    }
}
```


### [paddle.fluid.layers.log](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html#permalink-95-log)
```python
paddle.fluid.layers.log(
    x,
    name=None
)
```  

### 功能差异
#### 计算机制

Caffe：计算公式如下，  
$$
y=\begin{cases}
ln(shift+scale \times x),\quad base\leq 0 \\\\
log_{base}(shift+scale \times x),\quad base>0
\end{cases}
$$               
             
PaddlePaddle：计算公式如下，
$$y=ln(x)$$
