# 添加API映射方式
需要添加的API映射有5种情况，本文档将对5种情况进行分别介绍，而需要修改的文件在[x2paddle/code_convertor/pytorch](./x2paddle/code_convertor/pytorch)，具体文件如下所示：
> .  
> ├── api_mapper  
> │   ├── \_\_init\_\_.py  
> │   ├── learning_rate_scheduler.py  
> │   ├── nn.py  
> │   ├── ops.py  
> │   ├── torchvision.py  
> │   └── utils.py  
> ├── mapper.py  
> └── torch2paddle  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── container.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── device.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── \_\_init\_\_.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── io.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── layer.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── nn_functional.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── nn_utils.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── ops.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── optimizer.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── tensor.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── varbase.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── vision_transforms.py  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── vision_utils.py  

其中，mapper.py的作用是存放PyTorch到Paddle的API名字映射，以及进行参数映射所需调用的类；api_mapper文件夹下的每个文件存放没每个类别API中参数需要进行的映射处理， learning_rate_scheduler.py中是学习率类API， nn.py中是组网、loss等神经网络类API，ops.py中是Tensor处理类API，torchvision.py指视觉类API，后期如有新的类别，用户可以根据需求自行添加；torch2paddle存放各个类别需要重新实现的API。
### 情况一
当PyTorch中Tensor的类内函数不是Paddle中Tensor的内置函数，在运行代码时会出现如下错误：
``` shell
AttributeError: 'Tensor' object has no attribute 'XX'
```
该情况下需要将PyTorch中Tensor的类内函数注册为Paddle中Tensor的内置函数，注册方式为，在[x2paddle/code_convertor/pytorch/torch2paddle/tensor.py](./x2paddle/code_convertor/pytorch/torch2paddle/tensor.py)添加相应代码，以item类内函数为例，需要添加如下代码：
``` python
# 添加注册装饰器
@add_tensor_function
def item(self):
    # 实现item方法的对应功能
    return self.numpy()[0]
```
当PyTorch中Tensor的类内函数是Paddle中Tensor的内置函数，命名一致，但实现功能不一致，也需要重新实现该类内函数，以reshape类内函数为例，需要添加的代码如下：
``` python
# 对原始的reshape进行重命名，此处添加"_tmp"，防止与其他类内函数重名
reshape_tmp = partial(paddle.Tensor.reshape)
# 添加注册装饰器
@add_tensor_function
def reshape(self, *shape):
    # 实现reshape方法的对应功能
    return reshape_tmp(self, shape)
```
### 情况二
当PyTorch中Layer的类内函数不是Paddle中Layer的内置函数，在运行代码时会出现如下错误：
``` shell
AttributeError: 'Layer' object has no attribute 'XX'
```
该情况下需要将PyTorch中Layer的类内函数注册为Paddle中Layer的内置函数，注册方式为，在[x2paddle/code_convertor/pytorch/torch2paddle/layer.py](./x2paddle/code_convertor/pytorch/torch2paddle/layer.py)添加相应代码，以apply类内函数为例，需要添加如下代码：
``` python
# 添加注册装饰器
@add_layer_function
def apply(self, func):
    # 实现apply方法的对应功能
    func(self)
```
当PyTorch中Layer的类内函数是Paddle中Layer的内置函数，命名一致，但实现功能不一致，也需要重新实现该类内函数，以train类内函数为例，需要添加的代码如下：
``` python
# 对原始的train进行重命名，此处添加"_tmp"，防止与其他类内函数重名
train_tmp = partial(paddle.nn.Layer.train)
# 添加注册装饰器
@add_layer_function
def train(self, mode=True):
    # 实现train方法的对应功能
    return train_tmp(self)
```
### 情况三
当PyTorch的API与Paddle的API使用方式及参数一致，只是命名方式不一致时，直接在[x2paddle/code_convertor/pytorch/mapper.py](./x2paddle/code_convertor/pytorch/mapper.py)中对应的MAPPER中添加PyTorch API的字符串以及Paddle API的字符串，无需添加进行参数映射所需调用的类，具体实现如下：
``` python
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

### 情况四
在Paddle中可以找到与PyTorch功能相似的API，但PyTorch的API与Paddle的API命名方式、部分使用方式及参数不一致，同时Paddle可以通过参数替换以及简单添加几行多余操作实现时，使用该方式进行映射，主要有以下几个步骤：
1. 在[x2paddle/code_convertor/pytorch/mapper.py](./x2paddle/code_convertor/pytorch/mapper.py)中对应的MAPPER中添加PyTorch API的字符串以及Paddle API的字符串、参数映射等操作所需调用的类，具体实现如下：
``` python
# key为PyTorch API字符串；
# value为列表，由Paddle API字符串和参数映射等操作所需调用的类组合而成。
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

2. 在[x2paddle/code_convertor/pytorch/api_mapper/](./x2paddle/code_convertor/pytorch/api_mapper)文件夹中找到对应的文件并在其中添加参数映射等操作所需调用的类，以`torch.matmul`和`paddle.matmul`的映射为例，需要添加的类如下所示：
``` python
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
            return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        else:
            # 根据不同的可变参数更新字典参数
            same_attr_count = 2
            if len(self.args) > same_attr_count:
                new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args, same_attr_count)
                self.kwargs.update(new_kwargs)
                self.args = self.args[:same_attr_count]
            return self.convert_to_paddle()
```
其中，类中用到的几个方法介绍如下：
> rename_key(kwargs, old_key, new_key)：kwargs为PyTorch API的关键字参数，old_key为PyTorch API的关键字参数的key，new_key为Paddle API的关键字参数的key。作用：若old_key存在于kwargs，将old_key替换为new_key。
> delete_key(kwargs, old_key)：kwargs为PyTorch API的关键字参数，old_key为PyTorch API的关键字参数的key。作用：删除kwargs中的old_key。
> api_args2kwargs(pytorch_api_name, args, same_attr_count)：pytorch_api_name为PyTorch API字符串，args为PyTorch API的可变参数，same_attr_count为PyTorch和Paddle该API可变参数相同的数目。作用：将不同的可变参数转为字典参数。

`Mapper`基类的定义如下：
``` python
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
        """ 判断是否为可变参数或者关键字参数,
            若为可变参数或者关键字参数，则替换参数名。
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
参数映射等操作所需调用的类`FuncMatmul`重写了基类`Mapper`中的`process_attrs`、`delete_attrs`和`run`，最后`run`返回3个值，第一个值代表在paddle相关代码这一行之前需要添加的代码（为list），第二个值代表paddle相关代码（为str），第三行代表在paddle相关代码这一行之后需要添加的代码（为list）。

3. 当PyTorch API传入的是可变参数或关键字参数，参数映射等操作所需调用的类无法对参数进行处理，此时只能调用x2paddle封装的API，所以需要在[x2paddle/code_convertor/pytorch/torch2paddle/](./x2paddle/code_convertor/pytorch/torch2paddle)文件夹中找到对应的文件并在其中添加x2paddle API实现，其函数名或类名与torch2paddle_func_name中的命名一致，同样以`torch.matmul`和`paddle.matmul`的映射为例，其实现代码如下：
``` python 
def matmul(input, other, *, out=None):
    return paddle.matmul(input, other)
```

### 情况五
在Paddle中无法找到与PyTorch功能相似的API，需要自行实现该API，使用该方式进行映射，主要有以下几个步骤：
1.  在[x2paddle/code_convertor/pytorch/mapper.py](./x2paddle/code_convertor/pytorch/mapper.py)中对应的MAPPER中添加PyTorch API的字符串以及Paddle API的字符串，具体实现如下：
``` python
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

2. 在[x2paddle/code_convertor/pytorch/torch2paddle/](./x2paddle/code_convertor/pytorch/torch2paddle)文件夹中找到对应的文件并在其中添加x2paddle API实现，其函数名或类名与1.中字典 value值list的第一个值的命名一致，同样以`torch.utils.data.random_split`的实现为例，其实现代码如下：
``` python
def random_split(dataset, lengths, generator=None):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = paddle.randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
setattr(paddle.io, "random_split", random_split)
```
