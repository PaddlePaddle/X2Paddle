# PyTorch2Paddle

PyTorch2Paddle支持trace和script两种方式的转换，均是PyTorch动态图到Paddle动态图的转换，转换后的Paddle动态图运用动转静可转换为静态图模型。trace方式生成的代码可读性较强，较为接近原版PyTorch代码的组织结构；script方式不需要知道输入数据的类型和大小即可转换，使用上较为方便，但目前PyTorch支持的script代码方式有所限制，所以支持转换的代码也有所限制。用户可根据自身需求，选择转换方式。

## 环境依赖

python == 2.7 | python >= 3.5  
paddlepaddle >= 2.0.0  
pytorch：torch >=1.5.0 (script方式暂不支持1.7.0)

**使用trace方式需安装以下依赖**
pandas
treelib

## 使用方式

``` python
from x2paddle.convert import pytorch2paddle
pytorch2paddle(module=torch_module,
               save_dir="./pd_model",
               jit_type="trace",
               input_examples=[torch_input])
# module (torch.nn.Module): PyTorch的Module。
# save_dir (str): 转换后模型的保存路径。
# jit_type (str): 转换方式。默认为"trace"。
# input_examples (list[torch.tensor]): torch.nn.Module的输入示例，list的长度必须与输入的长度一致。默认为None。
```

**注意：** 当jit_type为"trace"时，input_examples不可为None，转换后自动进行动转静；
          当jit_type为"script"时"，input_examples不为None时，才可以进行动转静。

## 使用示例

``` python
import torch
import numpy as np
from torchvision.models import AlexNet
from torchvision.models.utils import load_state_dict_from_url
# 构建输入
input_data = np.random.rand(1, 3, 224, 224).astype("float32")
# 获取PyTorch Module
torch_module = AlexNet()
torch_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
torch_module.load_state_dict(torch_state_dict)
# 设置为eval模式
torch_module.eval()
# 进行转换
from x2paddle.convert import pytorch2paddle
pytorch2paddle(torch_module,
               save_dir="pd_model_trace",
               jit_type="trace",
               input_examples=[torch.tensor(input_data)])
```
