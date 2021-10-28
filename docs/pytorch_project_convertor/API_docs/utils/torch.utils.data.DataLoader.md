## torch.utils.data.DataLoader
### [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)
```python
torch.utils.data.DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            sampler=None,
                            batch_sampler=None,
                            num_workers=0,
                            collate_fn=None,
                            pin_memory=False,
                            drop_last=False,
                            timeout=0,
                            worker_init_fn=None,
                            multiprocessing_context=None,
                            generator=None,
                            prefetch_factor=2,
                            persistent_workers=False)
```

### [paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)
```python
paddle.io.DataLoader(dataset,
                     feed_list=None,
                     places=None,
                     return_list=False,
                     batch_sampler=None,
                     batch_size=1,
                     shuffle=False,
                     drop_last=False,
                     collate_fn=None,
                     num_workers=0,
                     use_buffer_reader=True,
                     use_shared_memory=False,
                     timeout=0,
                     worker_init_fn=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| sampler  | -        | 表示数据集采集器，PaddlePaddle无此参数。  |
| prefetch_factor  | -        | 表示每个worker预先加载的数据数量，PaddlePaddle无此参数。  |
| persistent_workers  | -        | 表示数据集使用一次后，数据加载器将会不会关闭工作进程，PaddlePaddle无此参数。  |
| generator        | -            | 用于采样的伪随机数生成器，PaddlePaddle无此参数。                   |
| pin_memory        | -            | 表示数据最开始是属于锁页内存，PaddlePaddle无此参数。                   |
| -        | feed_list      | 表示feed变量列表，PyTorch无此参数。                   |
| -        | use_buffer_reader      | 表示是否使用缓存读取器，PyTorch无此参数。                   |
| -        | use_shared_memory      | 表示是否使用共享内存来提升子进程将数据放入进程间队列的速度，PyTorch无此参数。                   |

### 功能差异
#### 自定义数据采集器
***PyTorch***：可通过设置`sampler`自定义数据采集器。  
***PaddlePaddle***：PaddlePaddle无此功能，可使用如下代码自定义一个DataLoader实现该功能。
```python
class DataLoader(paddle.io.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn=None,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False
            
        super().__init__(
            dataset,
            feed_list=None,
            places=None,
            return_list=return_list,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=True,
            use_shared_memory=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn)
        if sampler is not None:
            self.batch_sampler.sampler = sampler
```
