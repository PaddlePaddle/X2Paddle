# 添加API映射方式

在3种情况下需要添加的API映射，本文档将对添加方式逐一进行介绍，3种情况如下表所示：

|                      | 对应情况                                                     |
| -------------------- | ------------------------------------------------------------ |
| [情况1](#situation1) | 在运行代码时出现错误：`AttributeError: 'Tensor' object has no attribute 'XX'`。 |
| [情况2](#situation2) | 在运行代码时出现错误：`AttributeError: 'Layer' object has no attribute 'XX'`。 |
| [情况3](#situation3) | 在转换代码时出现提示：`Can not convert the file XX.py. The unsupported packages or operators are: [torch.nn.Tanh, torch.nn.utils.spectral_norm].`。<br/>[3.1](#situation3.1) PaddlePaddle存在对应API，功能完全一致，参数一致。 <br/>[3.2](#situation3.2) PaddlePaddle存在对应API，功能基本一致，参数不一致。 <br/>[3.3](#situation3.3) PaddlePaddle不存在对应API。 |

需要修改的文件在[x2paddle/project_convertor/pytorch](../../x2paddle/project_convertor/pytorch)中，具体文件如下所示：

> .  
> |── api_mapper     # 存放映射处理相关操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── \_\_init\_\_.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── learning_rate_scheduler.py    # 学习率类API映射操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── nn.py    # 组网、损失相关类API映射操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── ops.py    # paddle.Tensor处理类API映射操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── torchvision.py    # 图像处理相关的API映射操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── utils.py    # 基础操作  
> |── mapper.py    # 存放映射关系  
> └── torch2paddle    # 存放需要重新封装实现的API   
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|──\_\_init\_\_.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── device.py    # 实现设备相关的操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── io.py    # 实现数据相关的操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── layer.py    # 实现paddle.nn.Layer类内方法/属性的操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── nn_functional.py    # 实现组网OP的操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── nn_utils.py    # 实现组网参数相关的操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── ops.py    # 实现Tensor处理OP的操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── optimizer.py    # 实现优化相关的操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── tensor.py    # 实现paddle.Tensor类内方法/属性操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── varbase.py    # 实现paddle.Tensor取值的操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── vision_transforms.py    # 实现视觉transform的操作  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── vision_utils.py    # 实现视觉基础的操作  

***[注意]*** 添加完映射后，需要重新安装X2Paddle：
```
cd X2Paddle
rm -rf bulid dist x2paddle.egg-info
pip uninstall x2paddle
python setup.py install
```

### <span id="situation1">情况1</span>

该情况出现的原因为paddle.Tensor缺乏类内方法/属性，因此需要将torch.Tensor的类内方法/属性注册为paddle.Tensor的类内方法/属性，在[x2paddle/project_convertor/pytorch/torch2paddle/tensor.py](../../x2paddle/project_convertor/pytorch/torch2paddle/tensor.py)添加相应代码。以item类内方法为例，PyTorch中item方法的作用是提取Scalar中的数值，以避免耗费内存和计算量，因此需要添加如下代码：

```python
# 添加注册装饰器
@add_tensor_function
def item(self):
    # 实现item方法的对应功能
    return self.numpy()[0]
```

当torch.Tensor的类内方法/属性与paddle.Tensor的内置方法/属性命名一致，但实现功能不一致，也需要重新实现该类内方法/属性。以reshape类内方法为例，PyTorch传入的为可变参数，而PaddlePaddle出入的参数为一个list，因此需要添加的代码如下：

```python
# 对原始的reshape进行重命名，此处添加"_tmp"，防止与其他类内函数重名
reshape_tmp = partial(paddle.Tensor.reshape)
# 添加注册装饰器
@add_tensor_function
def reshape(self, *shape):
    # 实现reshape方法的对应功能
    return reshape_tmp(self, shape)
```

### <span id="situation2">情况2</span>

该情况出现的原因为paddle.nn.Layer缺乏类内方法/属性，因此需要将torch.nn.Module的类内方法/属性注册为paddle.nn.Layer的类内方法/属性，在[x2paddle/project_convertor/pytorch/torch2paddle/layer.py](../../x2paddle/project_convertor/pytorch/torch2paddle/layer.py)添加相应代码。以cuda类内方法为例，PyTorch的网络可以设置运行的的设备为cuda，而PaddlePaddle则不需要此操作，因此需要添加如下代码返回原网络即可：

```python
# 添加注册装饰器
@add_layer_function
def cuda(self):
    return self
```

当torch.nn.Module的类内方法/属性与paddle.nn.Layer的内置方法/属性命名一致，但实现功能不一致，也需要重新实现该类内方法/属性。以train类内方法为例，PyTorch可以设置train的模式是train还是eval，PaddlePaddle则需要组合实现，因此需要添加的代码如下：

```python
# 对原始的train进行重命名，此处添加"_tmp"，防止与其他类内函数重名
train_tmp = partial(paddle.nn.Layer.train)
# 添加注册装饰器
@add_layer_function
def train(self, mode=True):
    # 实现train方法的对应功能
    if mode:
        return train_tmp(self)
    else:
        return paddle.nn.Layer.eval(self)
```

### <span id="situation3">情况3</span>

### <span id="situation3.1">3.1</span> PaddlePaddle存在对应API，功能完全一致，参数一致

该情况直接在[x2paddle/project_convertor/pytorch/mapper.py](../../x2paddle/project_convertor/pytorch/mapper.py)中对应的MAPPER中添加PyTorch API的字符串以及Paddle API的字符串，无需添加进行参数映射所需调用的类，具体实现如下：

```python
# key为PyTorch API字符串；
# value为列表，由Paddle API字符串和None组合而成。
...
NN_MAPPER = {
      ...
      "torch.nn.Sequential":
         ["paddle.nn.Sequential", None],
      "torch.nn.utils": 
         ["paddle.nn.utils", None],
      ...
}
...
```

### <span id="situation3.2">3.2</span> PaddlePaddle存在对应API，功能基本一致，参数不一致

该情况需要完成以下几个步骤：

***步骤1*** 在[x2paddle/project_convertor/pytorch/mapper.py](.../../x2paddle/project_convertor/pytorch/mapper.py)中对应的MAPPER中添加PyTorch API的字符串以及Paddle API的字符串、映射处理类，具体实现如下：

```python
# key为PyTorch API字符串；
# value为列表，由Paddle API字符串和参映射处理类组合而成。
...
NN_MAPPER = {
      ...
      "torch.nn.Conv2d": 
          ["paddle.nn.Conv2D", Conv2DModuleMapper],
       ...
       "torch.nn.functional.relu": 
          ["paddle.nn.functional.relu", ReluFuncMapper],
       ...
       }
...
# 类名中的Module和Func分别代表torch.nn中的类和torch.nn中的方法，
# 不带Module和Func则非torch.nn中的操作。
```

***步骤2*** 在[x2paddle/project_convertor/pytorch/api_mapper/](../../x2paddle/project_convertor/pytorch/api_mapper)文件夹中找到对应的文件并在其中添加映射处理类，类型中用户需要重写process_attrs、delete_attrs、check_attrs以及run这三个函数，其中run只需要修改对应的x2paddle封装的API命名即可。以`torch.matmul`和`paddle.matmul`的映射为例，二者的参数名不一致，因此需要添加的代码如下所示：

```python
class MatmulMapper(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        """ 更新参数。
        """
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "other", "y")
        
    def delete_attrs(self):
        """ 删除参数。
        """
        delete_key(self.kwargs, "out")
        
    def check_attrs(self):
        """ 确认参数的值。
        """
        pass
        
    def run(self):
        if self.rename_func_name("x2paddle.torch2paddle.matmul"):
            # 作用：当出现可变参数或关键字参数，无法对参数进行处理；
            # 需要根据x2paddle封装的对应API命名生成代码(x2paddle封装的对应API相关代码在步骤3中实现)
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # 作用：将paddle与pytorch不同的可变参数替换成字典参数，并生成相应代码
            self.convert_args2kwargs()
            return self.convert_to_paddle()
```

其中，使用到的几个方法介绍如下：

| 方法                                 | 参数                                                         | 作用                                            |
| ------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------- |
| rename_key(kwargs, old_key, new_key) | kwargs：PyTorch API的关键字参数； old_key：PyTorch API的关键字参数的key； new_key：Paddle API的关键字参数的key。 | 若old_key存在于kwargs，将old_key替换为new_key。 |
| delete_key(kwargs, old_key)          | kwargs：PyTorch API的关键字参数；old_key：PyTorch API的关键字参数的key。 | 删除kwargs中的old_key。                         |



***步骤3*** 当PyTorch API传入的是可变参数或关键字参数，映射处理类无法对参数进行处理，此时只能调用x2paddle封装的API，所以需要在[x2paddle/project_convertor/pytorch/torch2paddle/](../../x2paddle/project_convertor/pytorch/torch2paddle)文件夹中找到对应的文件并在其中添加x2paddle API实现，其函数名或类名与步骤2中的`torch2paddle_func_name`命名一致，同样以`torch.matmul`和`paddle.matmul`的映射为例，其实现代码如下：

```python 
def matmul(input, other, *, out=None):
    return paddle.matmul(input, other)
```

### <span id="situation3.3">3.3</span> PaddlePaddle不存在对应API

### 3.3.1 API代码为必要代码

当前API在代码中必须存在，需要添加转换，因此要完成以下2个步骤：

***步骤1*** 在[x2paddle/project_convertor/pytorch/mapper.py](../../x2paddle/project_convertor/pytorch/mapper.py)中对应的MAPPER中添加PyTorch API的字符串以及Paddle API的字符串，具体实现如下：

```python
# key为PyTorch API字符串；
# value为列表，由x2paddle自行实现API字符串和None组合而成。
...
UTILS_MAPPER = {
      ...
      "torch.utils.data.random_split": 
          ["x2paddle.torch2paddle.random_split", None],
      ...
      "torch.utils.data.ConcatDataset": 
          ["x2paddle.torch2paddle.ConcatDataset", None]
       ...
       }
...
```

***步骤2*** 在[x2paddle/project_convertor/pytorch/torch2paddle/](../../x2paddle/project_convertor/pytorch/torch2paddle)文件夹中找到对应的文件并在其中添加x2paddle API实现，其函数名或类名与步骤1中字典 value值中list的第一个值一致，以`torch.utils.data.random_split`的实现为例，其作用为划分数据集，因此需要添加的代码如下所示：

```python
def random_split(dataset, lengths, generator=None):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = paddle.randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
setattr(paddle.io, "random_split", random_split)
```

### 3.3.2  API代码为不必要代码

当前API为PaddlePaddle不需要的代码，应进行删除，因此需要在[x2paddle/project_convertor/pytorch/mapper.py](../../x2paddle/project_convertor/pytorch/mapper.py)中REMOVE_API中添加需要去除的PyTorch API，具体实现如下：

```python
REMOVE_API =["torch.backends.cudnn",
             "torch.backends.cudnn.benchmark"]
```

### 3.3.3  API代码为可替换代码

若当前API可用其他PyTorch API`torch.YY`（`torch.YY`在[已支持映射列表](./supported_API.md)中）代替且替换后精度影响不大，可在原PyTorch代码中将当前API替换为`torch.YY`，再进行转换。

