# X2paddle API


## 目录
* [x2paddle.convert.tf2paddle](#1)
* [x2paddle.convert.caffe2paddle](#2)
* [x2paddle.convert.onnx2paddle](#3)
* [x2paddle.convert.pytorch2paddle](#4)


## <h2 id="1">x2paddle.convert.tf2paddle</h2>

```python
x2paddle.convert.tf2paddle(model_path, save_dir, define_input_shape=False, convert_opt=False, valid_places="arm", model_type="naive_buffer")
```

> 转换TensorFlow模型。

> **参数**
>
> > - **model_path** (str): TensorFlow pb模型路径
> > - **save_dir** (str): 转换后模型保存路径
> > - **define_input_shape** (bool): 是否指定输入大小，默认为False
> > - **convert_opt** (bool): 是否使用opt工具，默认为False
> > - **valid_places** (str): 指定转换类型，默认为arm
> > - **model_type** (str): 指定模型转化类型，默认为naive_buffer


## <h2 id="2">x2paddle.convert.caffe2paddle</h2>

```python
x2paddle.convert.caffe2paddle(proto, weight, save_dir, caffe_proto, convert_opt=False, valid_places="arm", model_type="naive_buffer")
```

> 转换Caffe模型。

> **参数**
>
> > - **proto** (str): caffe模型的prototxt文件
> > - **weight** (str): caffe模型的权重文件
> > - **save_dir** (str): 转换后模型保存路径
> > - **caffe_proto** (str): 可选：caffe模型的caffe proto文件编译的.py文件
> > - **convert_opt** (bool): 是否使用opt工具，默认为False
> > - **valid_places** (str): 指定转换类型，默认为arm
> > - **model_type** (str): 指定模型转化类型，默认为naive_buffer


## <h2 id="3">x2paddle.convert.onnx2paddle</h2>

```python
x2paddle.convert.onnx2paddle(model_path, save_dir, convert_opt=False, valid_places="arm", model_type="naive_buffer")
```

> 转换ONNX模型。

> **参数**
>
> > - **model_path** (str): TensorFlow pb模型路径
> > - **save_dir** (str): 转换后模型保存路径
> > - **convert_opt** (bool): 是否使用opt工具，默认为False
> > - **valid_places** (str): 指定转换类型，默认为arm
> > - **model_type** (str): 指定模型转化类型，默认为naive_buffer


## <h2 id="4">x2paddle.convert.pytorch2paddle</h2>

```python
x2paddle.convert.pytorch2paddle(module, save_dir, jit_type="trace", input_examples=None, convert_opt=False, valid_places="arm", model_type="naive_buffer")
```

> 转换Pytorch模型。

> **参数**
>
> > - **module** (torch.nn.Module): PyTorch的Module
> > - **save_dir** (str): 转换后模型保存路径
> > - **jit_type** (str): 转换方式。目前有两种:trace和script,默认为trace
> > - **input_examples** (list[torch.tensor]): torch.nn.Module的输入示例，list的长度必须与输入的长度一致。默认为None。
> > - **convert_opt** (bool): 是否使用opt工具，默认为False
> > - **valid_places** (str): 指定转换类型，默认为arm
> > - **model_type** (str): 指定模型转化类型，默认为naive_buffer
