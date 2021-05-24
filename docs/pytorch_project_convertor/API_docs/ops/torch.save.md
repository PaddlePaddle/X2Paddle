## torch.save
### [torch.save](https://pytorch.org/docs/stable/generated/torch.save.html?highlight=save#torch.save)

```python
torch.save(obj,
           f,
           pickle_module=pickle,
           pickle_protocol=2)
```

### [paddle.save](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/save_cn.html#save)

```python
paddle.save(obj, path, pickle_protocol=2)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| f        | path          | 表示存储的路径。                   |
| pickle_module          | -        | 表示用于pickling元数据和对象的模块，PaddlePaddle无此参数。                       |


### 功能差异

#### 存储类型
***PyTorch***：可存储到文件或者内存中的写缓冲区(例如`io.BytesIO`、`io.StringIO`)。  
***PaddlePaddle***：只能存储到文件中。

#### 存储内容
***PyTorch***：可以存储`torch.Tensor`、`torch.nn.Module`、优化器等多个类型的数据。  
***PaddlePaddle***：只能存储`paddle.nn.Layer`、优化器这两个类型的数据。


### 代码示例
``` python
# PyTorch示例：
x = torch.tensor([0, 1, 2, 3, 4])
buffer = io.BytesIO()
torch.save(x, buffer)
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([0, 1, 2, 3, 4])
padle.save(x, "tensor.pdiparams")
# 报错：
# NotImplementedError: Now only supports save state_dict of Layer or Optimizer, expect dict, but received <class 'paddle.VarBase'>.
emb = paddle.nn.Embedding(10, 10)
layer_state_dict = emb.state_dict()
paddle.save(layer_state_dict, "emb.pdparams")
# 正常保存
```
