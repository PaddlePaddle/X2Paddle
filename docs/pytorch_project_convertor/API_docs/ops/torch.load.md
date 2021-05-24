## torch.load
### [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html?highlight=load#torch.load)

```python
torch.load(f,
           map_location=None,
           pickle_module=pickle,
           **pickle_load_args)
```

### [paddle.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/load_cn.html#load)

```python
paddle.load(path, **configs)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| pickle_module          | -        | 表示用于unpickling元数据和对象的模块，PaddlePaddle无此参数。                       |
| map_location        | -            | 表示加载模型的位置，PaddlePaddle无此参数。                   |


### 功能差异

#### 加载类型
***PyTorch***：可从文件或者内存中的读缓冲区(例如`io.BytesIO`、`io.StringIO`)中加载。  
***PaddlePaddle***：只能从文件中加载。

#### 加载内容
***PyTorch***：可以加载`torch.Tensor`、`torch.nn.Module`、优化器等多个类型的数据。  
***PaddlePaddle***：只能加载`paddle.nn.Layer`、优化器这两个类型的数据。


### 代码示例
``` python
# PyTorch示例：
torch.load('tensors.pt', map_location=torch.device('cpu'))
```

``` python
# PaddlePaddle示例：
load_layer_state_dict = paddle.load("emb.pdparams")
```
