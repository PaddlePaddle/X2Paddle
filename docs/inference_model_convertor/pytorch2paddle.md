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

```python
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

**注意:**
- jit_type为"trace"时，input_examples不可为None，转换后自动进行动转静；
- jit_type为"script"时，当input_examples为None时，只生成动态图代码；当input_examples不为None时，才能自动动转静。

## 使用示例

### Trace 模式

```python
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

### 动态 shape 导出

#### 方式一：PyTorch->ONNX->Paddle

```python
import torch
from torchvision.models import AlexNet
from torchvision.models.utils import load_state_dict_from_url

# 获取PyTorch Module
torch_module = AlexNet()
torch_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
torch_module.load_state_dict(torch_state_dict)
# 设置为eval模式
torch_module.eval()
input_names = ["input_0"]
output_names = ["output_0"]

x = torch.randn((1, 3, 224, 224))
y = torch.randn((1, 1000))

torch.onnx.export(torch_module, x, 'model.onnx', opset_version=11, input_names=input_names,
                  output_names=output_names, dynamic_axes={'input_0': [0], 'output_0': [0]})
```

导出 ONNX 动态 shape 模型，更多细节参考[相关文档](https://pytorch.org/docs/stable/onnx.html?highlight=onnx%20export#torch.onnx.export)

然后通过 X2Paddle 命令导出 Paddle 模型

```shell
x2paddle --framework=onnx --model=model.onnx --save_dir=pd_model_dynamic
```

#### 方式二：手动动转静

在自动生成的 x2paddle_code.py 中添加如下代码：

```python
def main(x0):
    # There are 0 inputs.
    paddle.disable_static()
    params = paddle.load('model.pdparams')
    model = AlexNet()
    model.set_dict(params)
    model.eval()
    ## convert to jit
    sepc_list = list()
    sepc_list.append(
            paddle.static.InputSpec(
                shape=[-1, 3, 224, 224], name="x0", dtype="float32"))
    static_model = paddle.jit.to_static(model, input_spec=sepc_list)
    paddle.jit.save(static_model, "pd_model_trace/inference_model/model")
    out = model(x0)
    return out
```

然后运行 main 函数导出动态 shape 模型
