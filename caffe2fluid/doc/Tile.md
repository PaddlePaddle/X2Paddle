## Tile


### [Tile](http://caffe.berkeleyvision.org/tutorial/layers/tile.html)
```
layer {
    name: "tile"
    type: "Tile"
    bottom: "data"
    top: "concat"
    tile_param{
        axis: 1
        tiles: 2
    }
}
```


### [paddle.fluid.layers.concat](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#permalink-70-expand)
```python
paddle.fluid.layers.concat(
    x, 
    expand_times, 
    name=None
)
```  

### 功能差异
#### 输入参数
Caffe：只能在一个维度上进行复制。                    
PaddlePaddle：`expand_times`为一个list或tuple，它存放的是每个维度复制的倍数。
