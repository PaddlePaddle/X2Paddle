# X2paddle API

## 目录

* [x2paddle.convert.tf2paddle](#1)
* [x2paddle.convert.caffe2paddle](#2)
* [x2paddle.convert.onnx2paddle](#3)
* [x2paddle.convert.pytorch2paddle](#4)

TensorFlow、Caffe、ONNX以及PyTorch模型转换API如下：

## <h2 id="1">x2paddle.convert.tf2paddle</h2>

```python
x2paddle.convert.tf2paddle(model_path, save_dir, define_input_shape=False, convert_to_lite=False, lite_valid_places="arm", lite_model_type="naive_buffer")
```

> 转换TensorFlow模型。

> **参数**
>
> > - **model_path** (str): TensorFlow pb模型路径
> > - **save_dir** (str): 转换后模型保存路径
> > - **define_input_shape** (bool): 是否指定输入大小，默认为False
> > - **convert_to_lite** (bool): 是否使用opt工具转成Paddle-Lite支持格式，默认为False
> > - **lite_valid_places** (str): 指定转换类型，可以同时指定多个backend(以逗号分隔)，opt将会自动选择最佳方式，默认为arm
> > - **lite_model_type** (str): 指定模型转化类型，目前支持两种类型：protobuf和naive_buffer，默认为naive_buffer

## <h2 id="2">x2paddle.convert.caffe2paddle</h2>

```python
x2paddle.convert.caffe2paddle(proto_file, weight_file, save_dir, caffe_proto, convert_to_lite=False, lite_valid_places="arm", lite_model_type="naive_buffer")
```

> 转换Caffe模型。

> **参数**
>
> > - **proto_file** (str): caffe模型的prototxt文件
> > - **weight_file** (str): caffe模型的权重文件
> > - **save_dir** (str): 转换后模型保存路径
> > - **caffe_proto** (str): 可选：由caffe.proto编译成caffe_pb2.py文件的存放路径，当存在自定义Layer时使用，默认为None
> > - **convert_to_lite** (bool): 是否使用opt工具转成Paddle-Lite支持格式，默认为False
> > - **lite_valid_places** (str): 指定转换类型，可以同时指定多个backend(以逗号分隔)，opt将会自动选择最佳方式，默认为arm
> > - **lite_model_type** (str): 指定模型转化类型，目前支持两种类型：protobuf和naive_buffer，默认为naive_buffer

## <h2 id="3">x2paddle.convert.onnx2paddle</h2>

```python
x2paddle.convert.onnx2paddle(model_path, save_dir, convert_to_lite=False, lite_valid_places="arm", lite_model_type="naive_buffer")
```

> 转换ONNX模型。

> **参数**
>
> > - **model_path** (str): TensorFlow pb模型路径
> > - **save_dir** (str): 转换后模型保存路径
> > - **convert_to_lite** (bool): 是否使用opt工具转成Paddle-Lite支持格式，默认为False
> > - **lite_valid_places** (str): 指定转换类型，可以同时指定多个backend(以逗号分隔)，opt将会自动选择最佳方式，默认为arm
> > - **lite_model_type** (str): 指定模型转化类型，目前支持两种类型：protobuf和naive_buffer，默认为naive_buffer

## <h2 id="4">x2paddle.convert.pytorch2paddle</h2>

```python
x2paddle.convert.pytorch2paddle(module, save_dir, jit_type="trace", input_examples=None, enable_code_optim=True, convert_to_lite=False, lite_valid_places="arm", lite_model_type="naive_buffer")
```

> 转换Pytorch模型。

> **参数**
>
> > - **module** (torch.nn. Module): PyTorch的Module
> > - **save_dir** (str): 转换后模型保存路径
> > - **jit_type** (str): 转换方式。目前有两种:trace和script, 默认为trace
> > - **input_examples** (list[torch.tensor]): torch.nn. Module的输入示例，list的长度必须与输入的长度一致。默认为None
> > - **enable_code_optim** (bool): 转换后的代码是否进行优化, 默认为True
> > - **convert_to_lite** (bool): 是否使用opt工具转成Paddle-Lite支持格式，默认为False
> > - **lite_valid_places** (str): 指定转换类型，可以同时指定多个backend(以逗号分隔)，opt将会自动选择最佳方式，默认为arm
> > - **lite_model_type** (str): 指定模型转化类型，目前支持两种类型：protobuf和naive_buffer，默认为naive_buffer
