## 如何转换Caffe自定义Layer

本文档介绍如何将Caffe自定义Layer转换为PaddlePaddle模型中的对应实现, 用户可根据自己需要，添加代码实现自定义层，从而支持模型的完整转换。
目前，代码中已经提供了10个非官方op（不在[官网](http://caffe.berkeleyvision.org/tutorial/layers)上的op）的转换，这些op对应的Caffe实现源码如下：

| op | 该版本实现源码 |
|-------|--------|
| PriorBox | [code](https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp) |
| DetectionOutput | [code](https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/detection_output_layer.cpp) |
| ConvolutionDepthwise | [code](https://github.com/farmingyard/caffe-mobilenet/blob/master/conv_dw_layer.cpp) |
| ShuffleChannel | [code](https://github.com/farmingyard/ShuffleNet/blob/master/shuffle_channel_layer.cpp) |
| Permute | [code](https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/permute_layer.cpp) |
| Normalize | [code](https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/normalize_layer.cpp) |
| ROIPooling | [code](https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/layers/roi_pooling_layer.cpp) |
| Axpy | [code](https://github.com/hujie-frank/SENet/blob/master/src/caffe/layers/axpy_layer.cpp) |
| ReLU6 | [code](https://github.com/chuanqi305/ssd/blob/ssd/src/caffe/layers/relu6_layer.cpp) |
| Upsample | [code](https://github.com/eric612/MobileNet-YOLO/blob/master/src/caffe/layers/upsample_layer.cpp) |

添加代码实现自定义层的步骤如下：

***步骤一 下载代码***  
此处涉及修改源码，应先卸载x2paddle，并且下载源码，主要有以下两步完成：
```
pip uninstall x2paddle
pip install git+https://github.com/PaddlePaddle/X2Paddle.git@develop
```

***步骤二 编译caffe.proto***  
该步骤依赖protobuf编译器，其安装过程有以下两种方式：
> 选择一：pip install protobuf  (protobuf >= 3.6.0)
> 选择二：使用[官方源码](https://github.com/protocolbuffers/protobuf)进行编译

使用脚本./tools/compile.sh将caffe.proto（包含所需的自定义Layer信息）编译成我们所需的目标语言（Python）  
使用方式：
```
bash ./tools/compile.sh /home/root/caffe/src/caffe/proto
# /home/root/caffe/src/caffe/proto为caffe.proto的存放路径，生成的caffe_pb2.py也将保存在该路径下
```
将生成的caffe_pb2.py替换x2paddle/decoder下的caffe_pb2.py。

***步骤三 添加自定义Layer的实现代码***
> 【注意】若Caffe自定义layer与Paddle的op一一对应，使用方式一，否则使用方式二。

- 方式一：
1. 仿照./x2paddle/op_mapper/dygraph/caffe2paddle/caffe_op_mapper.py中的CaffeOpMapper类中的映射方法（输入为self和node），实现类似的映射方法，以下述的映射方法为例：
```python
def Permute(self, node):
    assert len(
        node.inputs) == 1, "The count of Permute node\'s input is not 1."
    input = self.graph.get_input_node(node, idx=0, copy=True)
    params = node.layer.permute_param
    order = list(params.order)  
    self.paddle_graph.add_layer(
        "paddle.transpose",
        inputs={"x": input.name},
        outputs=[node.layer_name],
        perm=order)
```
>需完成的步骤：
>    a. 获取Caffe Layer的属性，并对应转换为Paddle的属性。
>    b. 获取当前Layer的输入。
>    c. 使用self.paddle_graph.add_layer为PaddleGraph添加layer。其中，第一个参数代表Paddle的kernel；inputs是一个字典，用于存储paddle中的输入的key与其输入名字；outputs是一个列表，用于存储输出的名字；其余参数为属性对应关系。
2. 仿照./x2paddle/decoder/caffe_shape_inference.py中的shape_xx方法，实现获取当前Layer输出大小的函数，以下述方法为例：
```python
def shape_permute(layer, input_shape):
    order = layer.permute_param.order
    inshape = input_shape[0]
    output_shape = []
    order = list(order)
    for ii in order:
        assert ii < len(inshape), "invalid order for permute[%s]" % (name)
        output_shape.append(inshape[ii])
    return [output_shape]
```
>参数：
>    layer (caffe_pb2.LayerParameter): caffe的Layer，可用于获取当前Layer的属性。
>    input_shape (list): 其中每个元素代表该层每个输入数据的大小。


- 方式二：
1. 进入./x2paddle/op_mapper/dygraph/caffe2paddle/caffe_custom_layer，创建.py文件，例如mylayer.py
2. 仿照./x2paddle/op_mapper/dygraph/caffe2paddle/caffe_custom_layer中的其他文件，在mylayer.py中主要需要实现1个类，下面以roipooling.py为例分析代码：

```python
class ROIPooling(object):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        self.roipooling_layer_attrs = {
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "spatial_scale": spatial_scale}

    def __call__(self, x0, x1):
        slice_x1 = paddle.slice(input=x1, axes=[1],
                                starts=[1], ends=[5])
        out = fluid.layers.roi_pool(input=x0,
                                    rois=slice_x1,
                                    **self.roipooling_layer_attrs)
        return out
```

>\_\_init\_\_函数：用于初始化各个属性
>\_\_call\_\_函数：用于组合实现当前Layer的前向，输入为当前Layer所需要的输入


3. 仿照./x2paddle/op_mapper/dygraph/caffe2paddle/caffe_op_mapper.py中的CaffeOpMapper类中的映射方法（输入为self和node），实现类似的映射方法，以下述的映射方法为例：
```python
def ROIPooling(self, node):
      roipooling_name = name_generator("roipooling", self.nn_name2id)
      output_name = node.layer_name
      layer_outputs = [roipooling_name, output_name]
      assert len(
          node.inputs) == 2, "The count of ROIPooling node\'s input is not 2."
      input0 = self.graph.get_input_node(node, idx=0, copy=True)
      input1 = self.graph.get_input_node(node, idx=1, copy=True)
      inputs_dict = {}
      inputs_dict["x0"] = input0.name
      inputs_dict["x1"] = input1.name
      params = node.layer.roi_pooling_param
      layer_attrs = {
          "pooled_height": params.pooled_h,
          "pooled_width": params.pooled_w,
          "spatial_scale": params.spatial_scale}
      self.paddle_graph.add_layer(
          "custom_layer:ROIPooling",
          inputs=inputs_dict,
          outputs=layer_outputs,
          **layer_attrs)
```
>需完成的步骤：
>    a. 获取Caffe Layer的属性，并对应转换为Paddle的属性。
>    b. 获取当前Layer的输入。
>    c. 使用self.paddle_graph.add_layer为PaddleGraph添加layer。其中，第一个参数代表Paddle的kernel（此处kernel必须以“custom_layer:“开头）；inputs是一个字典，用于存储paddle中的输入的key与其输入名字；outputs是一个列表，用于存储输出的名字；其余参数为属性对应关系。

4. 仿照./x2paddle/decoder/caffe_shape_inference.py中的shape_xx方法，实现获取当前Layer输出大小的函数，以下述方法为例：
```python
def shape_roipooling(layer, input_shape):
  pooled_w = layer.roi_pooling_param.pooled_w
  pooled_h = layer.roi_pooling_param.pooled_h
  base_fea_shape = input_shapes[0]
  rois_shape = input_shapes[1]
  output_shape = base_fea_shape
  output_shape[0] = rois_shape[0]
  output_shape[2] = pooled_h
  output_shape[3] = pooled_w
  return [output_shape]

```
>参数：
>    layer (caffe_pb2.LayerParameter): caffe的Layer，可用于获取当前Layer的属性。
>    input_shape (list): 其中每个元素代表该层每个输入数据的大小。

***步骤四 运行转换代码***
```
# 在X2Paddle目录下安装x2paddle
python setup.py install
# 运行转换代码
x2paddle --framework=caffe
         --prototxt=deploy.proto
         --weight=deploy.caffemodel
         --save_dir=pd_model
         --caffe_proto=/home/root/caffe/src/caffe/proto/caffe_pb2.py
```
