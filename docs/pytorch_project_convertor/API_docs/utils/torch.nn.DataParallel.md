## torch.nn.DataParallel
### [torch.nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html?highlight=dataparallel#torch.nn.DataParallel)
```python
torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
```

### [paddle.DataParallel](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/DataParallel_cn.html#dataparallel)
```python
paddle.DataParallel(layers, strategy=None, comm_buffer_size=25, last_comm_buffer_size=1)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| module  | layers        | 需要通过数据并行方式执行的模型。  |
| device_ids  | -        | 表示训练在哪几块GPU上，PaddlePaddle无此参数。  |
| output_device | -   | 表示结果输出在哪一块GPU上，PaddlePaddle无此参数。  |
| dim  | -  | 表示哪一维度上的数据进行划分，PaddlePaddle无此参数。  |
| - | strategy |  PaddlePaddle即将废弃参数。 |
| - | comm_buffer_size |  它是通信调用（如NCCLAllReduce）时，参数梯度聚合为一组的内存大小（MB），PyTorch无此参数。 |
| - | last_comm_buffer_size |  它限制通信调用中最后一个缓冲区的内存大小（MB），PyTorch无此参数。 |

### 功能差异
#### 使用差异
***PyTorch***：在API中即可通过设置参数使用的GPU id。
***PaddlePaddle***：只能在启动代码时设置GPU id，设置方式如下：
> python -m paddle.distributed.launch –selected_gpus=0,1 demo.py
> 其中 demo.py 脚本的代码可以是下面的示例代码。
