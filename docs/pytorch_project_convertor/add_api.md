# 添加API映射方式

在5种情况下需要添加的API映射，本文档将对添加方式逐一进行介绍，5种情况如下表所示：

|                       | 对应情况                                                     |
| --------------------- | ------------------------------------------------------------ |
| [情况一](#situation1) | 在运行代码时出现错误：`AttributeError: 'Tensor' object has no attribute 'XX'`。 |
| [情况二](#situation2) | 在运行代码时出现错误：`AttributeError: 'Layer' object has no attribute 'XX'`。 |
| [情况三](#situation3) | 在转换代码时出现提示：`Can not convert the file XX.py. The no support Api are: [torchvision].`；同时，缺乏的PyTorch API，在PaddlePaddle中存在功能、使用方式及参数一致，只有命名不一致的API。 |
| [情况四](#situation4) | 在转换代码时出现提示：`Can not convert the file XX.py. The no support Api are: [torchvision].`；同时，缺乏的PyTorch API，在PaddlePaddle中存在功能一致，但使用方式或参数不一致，且命名不一致的API。 |
| [情况五](#situation5) | 在转换代码时出现提示：`Can not convert the file XX.py. The no support Api are: [torchvision].`；同时，缺乏的PyTorch API，在PaddlePaddle中不存在功能一致的API。 |

需要修改的文件在[x2paddle/project_convertor/pytorch](../../x2paddle/project_convertor/pytorch)中，具体文件如下所示：

> .  
> |── api_mapper     # 存放映射处理相关操作  
> |   |── \_\_init\_\_.py  
> |   |── learning_rate_scheduler.py    # 学习率类API映射操作  
> |   |── nn.py    # 组网、损失相关类API映射操作  
> |   |── ops.py    # paddle.Tensor处理类API映射操作  
> |   |── torchvision.py    # 图像处理相关的API映射操作  
> |  └── utils.py    # 基础操作  
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



### <span id="situation1">情况一</span>

该情况出现的原因为paddle.Tensor缺乏类内方法/属性，因此需要将torch.Tensor的类内方法/属性注册为paddle.Tensor的类内方法/属性，在[x2paddle/project_convertor/pytorch/torch2paddle/tensor.py](../../x2paddle/project_convertor/pytorch/torch2paddle/tensor.py)添加相应代码，以item类内方法为例，需要添加如下代码：

```python
# 添加注册装饰器
@add_tensor_function
def item(self):
    # 实现item方法的对应功能
    return self.numpy()[0]
```

当torch.Tensor的类内方法/属性与paddle.Tensor的内置方法/属性命名一致，但实现功能不一致，也需要重新实现该类内方法/属性，以reshape类内方法为例，需要添加的代码如下：

```python
# 对原始的reshape进行重命名，此处添加"_tmp"，防止与其他类内函数重名
reshape_tmp = partial(paddle.Tensor.reshape)
# 添加注册装饰器
@add_tensor_function
def reshape(self, *shape):
    # 实现reshape方法的对应功能
    return reshape_tmp(self, shape)
```

### <span id="situation2">情况二</span>

该情况出现的原因为paddle.nn.Layer缺乏类内方法/属性，因此需要将torch.nn.Module的类内方法/属性注册为paddle.nn.Layer的类内方法/属性，在[x2paddle/project_convertor/pytorch/torch2paddle/layer.py](../../x2paddle/project_convertor/pytorch/torch2paddle/layer.py)添加相应代码，以apply类内方法为例，需要添加如下代码：

```python
# 添加注册装饰器
@add_layer_function
def apply(self, func):
    # 实现apply方法的对应功能
    func(self)
```

当torch.nn.Module的类内方法/属性与paddle.nn.Layer的内置方法/属性命名一致，但实现功能不一致，也需要重新实现该类内方法/属性，以train类内方法为例，需要添加的代码如下：

```python
# 对原始的train进行重命名，此处添加"_tmp"，防止与其他类内函数重名
train_tmp = partial(paddle.nn.Layer.train)
# 添加注册装饰器
@add_layer_function
def train(self, mode=True):
    # 实现train方法的对应功能
    return train_tmp(self)
```

### <span id="situation3">情况三</span>

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

### <span id="situation4">情况四</span>

该情况需要完成以下几个步骤：

1. 在[x2paddle/project_convertor/pytorch/mapper.py](.../../x2paddle/project_convertor/pytorch/mapper.py)中对应的MAPPER中添加PyTorch API的字符串以及Paddle API的字符串、映射处理类，具体实现如下：

```python
# key为PyTorch API字符串；
# value为列表，由Paddle API字符串和参映射处理类组合而成。
...
NN_MAPPER = {
      ...
      "torch.nn.Conv2d": 
          ["paddle.nn.Conv2D", ClassConv2D],
       ...
       "torch.nn.functional.relu": 
          ["paddle.nn.functional.relu", FuncRelu],
       ...
       }
...
# 类名以Class或Func开始，Class代表Paddle API为一个类，Func代表Paddle API为一个方法。
```

2. 在[x2paddle/project_convertor/pytorch/api_mapper/](../../x2paddle/project_convertor/pytorch/api_mapper)文件夹中找到对应的文件并在其中添加映射处理类，以`torch.matmul`和`paddle.matmul`的映射为例，需要添加的类如下所示：

```python
class FuncMatmul(Mapper):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super().__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
    def process_attrs(self):
        rename_key(self.kwargs, "input", "x")
        rename_key(self.kwargs, "other", "y")
        
    def delete_attrs(self):
        delete_key(self.kwargs, "out")
        
    def run(self):
        if self.args_has_star("x2paddle.torch2paddle.matmul"):
            # 作用：当出现可变参数或关键字参数，无法对参数进行处理；
            # 需要根据x2paddle封装的对应API命名生成代码(x2paddle封装的对应API相关代码在步骤3中实现)
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # 作用：将paddle与pytorch不同的可变参数替换成字典参数，并生成相应代码
            same_attr_count = 2
            if len(self.args) > same_attr_count:
                new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
                self.kwargs.update(new_kwargs)
                self.args = self.args[:same_attr_count]
            return self.convert_to_paddle()
```

其中，使用到的几个方法介绍如下：

| 方法                                                     | 参数                                                         | 作用                                            |
| -------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------- |
| rename_key(kwargs, old_key, new_key)                     | kwargs：PyTorch API的关键字参数； old_key：PyTorch API的关键字参数的key； new_key：Paddle API的关键字参数的key。 | 若old_key存在于kwargs，将old_key替换为new_key。 |
| delete_key(kwargs, old_key)                              | kwargs：PyTorch API的关键字参数；old_key：PyTorch API的关键字参数的key。 | 删除kwargs中的old_key。                         |
| api_args2kwargs(pytorch_api_name, args, same_attr_count) | pytorch_api_name：PyTorch API命名； same_attr_count：paddle与pytorch的前same_attr_count个可变参数是一致的。 | 将paddle与pytorch不同的可变参数替换成字典参数   |

`Mapper`基类的定义如下：

```python
class Mapper(object):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
        self.target_name = target_name
        
    def process_attrs(self):
        """ 更新参数。
        """
        pass
     
    def delete_attrs(self):
        """ 删除参数。
        """
        pass
    
    def check_attrs(self):
        """ 确认参数的值。
        """
        pass
    
    def args_has_star(self, torch2paddle_func_name=None):
        """ 判断是否存在可变参数或者关键字参数,
            若为可变参数或者关键字参数，则替换参数名。
            Args:
                torch2paddle_func_name (str): 表示x2paddle封装的对应API名字。
        """
        if torch2paddle_func_name is not None and \
                (len(self.args) > 0 and isinstance(self.args[0], str) and self.args[0].startswith("*")) or \
                (len(self.args) > 1 and isinstance(self.args[-1], str) and self.args[-1].startswith("**")):
            self.func_name = torch2paddle_func_name
            return True
        else:
            return False
    
    def convert_to_paddle(self):
        """ 1. 通过执行check、process、delete转换为paddle的参数；
            2. 生成paddle相关代码。
        """
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        
    def run(self, torch2paddle_func_name=None):
        """ 如果存在可变参数或者关键字参数，直接替换函数名为x2paddle的API；
            反之，调用convert_to_paddle。
            Args:
                torch2paddle_func_name (str): 表示x2paddle封装的对应API名字。
        """
        if self.args_has_star(torch2paddle_func_name):
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            return self.convert_to_paddle()
```

映射处理类`FuncMatmul`重写了基类`Mapper`中的`process_attrs`、`delete_attrs`和`run`，最后`run`返回3个值，第一个值代表在替换的paddle相关代码这一行之前需要添加的代码（为list），第二个值代表替换的paddle相关代码（为str），第三行代表在替换的paddle相关代码这一行之后需要添加的代码（为list）。

3. 当PyTorch API传入的是可变参数或关键字参数，映射处理类无法对参数进行处理，此时只能调用x2paddle封装的API，所以需要在[x2paddle/project_convertor/pytorch/torch2paddle/](../../x2paddle/project_convertor/pytorch/torch2paddle)文件夹中找到对应的文件并在其中添加x2paddle API实现，其函数名或类名与步骤2中的`torch2paddle_func_name`命名一致，同样以`torch.matmul`和`paddle.matmul`的映射为例，其实现代码如下：

```python 
def matmul(input, other, *, out=None):
    return paddle.matmul(input, other)
```

### <span id="situation5">情况五</span>

该情况需要完成以下几个步骤：

1. 在[x2paddle/project_convertor/pytorch/mapper.py](../../x2paddle/project_convertor/pytorch/mapper.py)中对应的MAPPER中添加PyTorch API的字符串以及Paddle API的字符串，具体实现如下：

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

2. 在[x2paddle/project_convertor/pytorch/torch2paddle/](../../x2paddle/project_convertor/pytorch/torch2paddle)文件夹中找到对应的文件并在其中添加x2paddle API实现，其函数名或类名与步骤1中字典 value值中list的第一个值一致，以`torch.utils.data.random_split`的实现为例，其实现代码如下：
``` python
def random_split(dataset, lengths, generator=None):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = paddle.randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
setattr(paddle.io, "random_split", random_split)
```
