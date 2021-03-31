# PyTorch项目转换

支持将PyTorch代码及预训练模型转换为PaddlePaddle代码及预训练模型。

## 环境依赖
python >= 3.5  
paddlepaddle >= 2.0.1 或者 develop   
torch (版本由需转换的代码所需的运行环境决定)

## 使用方法
### 转换前操作
1. 去除TensorBoard相关的操作。

2. 将PyTorch中Tensor逐位逻辑与、或、异或运算操作符替换为对应的API的操作：
> | 替换为 torch.bitwise_or  
> & 替换为 torch.bitwise_and  
> ^ 替换为 toech.bitwise_xor   

``` python
# 原始代码：
pos_mask | neg_mask
# 替换后代码
torch.bitwise_or(pos_mask, neg_mask)
```

3. 若自定义的`DataSet`（用于加载数据模块，作为`torch.utils.data.DataLoader`的参数）未继承`torch.utils.data.Dataset`，则需要添加该继承关系。

```
# 原始代码
class VocDataset:
# 替换后代码
import torch
class VocDataset(torch.utils.data.Dataset):
```

4. 若预训练模型需要下载，去除下载预训练模型相关代码，在转换前将预训练模型下载至本地，并修改加载预训练模型参数相关代码的路径为预训练模型本地保存路径。

5. 若在数据预处理中出现Tensor与float型/int型对比大小，则需要将float型/int型修改为Tensor，例如下面代码为一段未数据预处理中一段代码，修改如下：
``` python
# 原始代码：
mask = best_target_per_prior < 0.5
# 替换后代码
threshold_tensor = torch.full_like(best_target_per_prior, 0.5)
mask = best_target_per_prior < threshold_tensor
```

### 转换
``` shell
x2paddle --convert_torch_project --project_dir=torch_project --save_dir=paddle_project --pretrain_model=model.pth
```
| 参数 | |
|----------|--------------|
|--convert_torch_project | 当前方式为对PyTorch Project进行转换 |
|--project_dir | PyTorch的项目路径 |
|--save_dir | 指定转换后项目的保存路径 |
|--pretrain_model | **[可选]**需要转换的预训练模型的路径(文件后缀名为“.pth”、“.pt”、“.ckpt”)或者包含预训练模型的文件夹路径，转换后的模型将将保在当前路径，后缀名为“.pdiparams” |


### 转换后操作
1. 若需要使用GPU，`x2paddle.torch2paddle.DataLoader`中的`num_workers`必须设置为0。

2. 修改自定义Dataset（继承自`paddle.io.Dataset`）中的`__getitem__`的返回值，若返回值中存在Tensor，需添加相应代码将Tensor修改为numpy。

```
# 原始代码
class VocDataset(paddle.io.Dataset):
    ...
    def __getitem__(self):
        ...
        return out1, out2
    ...
# 替换后代码
class VocDataset(paddle.io.Dataset):
    ...
    def __getitem__(self):
        ...
        if isinstance(out1, paddle.Tensor):
            out1 = out1.numpy()
        if isinstance(out2, paddle.Tensor):
            out2 = out2.numpy()
        return out1, out2
    ...
```

3. 若存在Tensor对比操作（包含==、!=、<、<=、>、>=操作符）,在对比操作符前添加对Tensor类型的判断，如果为bool型强转为int型，并在对比后转换回bool型。

```
# 原始代码（其中c_trg是Tensor）
c_trg = c_trg == 0
# 替换后代码
is_bool = False
if str(c_trg.dtype) == "VarType.BOOL":
    c_trg = c_trg.cast("int32")
    is_bool = True
c_trg = c_trg == 0
if is_bool:
    c_trg = c_trg.cast("bool")
```

4. 如若转换后的运行代码的入口为sh脚本文件去其中有预训练模型路径，应将其中的预训练模型的路径字符串中的“.pth”、“.pt”、“.ckpt”替换为“.pdiparams”。

***[注意]*** 转换前后相应操作可以参考[转换示例](./demo.md)

## Q&A
1. 出现如下提示如何处理？
> The no support Api are: [torchvision.transforms.RandomErasing, torchvision.transforms.functional, torchvision.transforms.RandomCrop.get_params, torch.all, torch.as_tensor].  

A：这一提示说明仍有API未支持转换，用户可自行添加相应API的支持，具体添加流程参照[添加示例](./add_api.md)，或及时提issue与我们联系。  

2. 运行时，出现DataLoader的报错异常，如何查找原因？  
A：  
步骤一：查看对应自定义Dataset中\_\_getiem\_\_的返回值是否为numpy；  
步骤二：如若当前的设备为GPU，是否未将`num_workers`设置为0；  
步骤三：查看图像预处理的transform中是否有使用出错。  

3. 当前是否支持torch.jit的转换？  
A：不支持。  
