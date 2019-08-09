## 如何转换Caffe自定义Layer

本文档介绍如何将Caffe自定义Layer转换为PaddlePaddle模型中对应的实现, 用户可根据自己需要，添加代码实现自定义层，从而支持模型的完整转换。

***步骤一 编译caffe.proto***    
使用脚本./tools/compile.sh将caffe.proto（包含所需的自定义Layer信息）编译成我们所需的目标语言（Python）         
使用方式：
```
bash ./toos/compile.sh /home/root/caffe/src/caffe/proto
# /home/root/caffe/src/caffe/proto为caffe.proto的存放路径，生成的caffe_pb2.py也将保存在该路径下
```

***步骤二 添加自定义Layer的实现代码***
- 进入./x2paddle/op_mapper/caffe_custom_layer，创建实现代码的文件，例如mylayer.py
- 仿照./x2paddle/op_mapper/caffe_custom_layer中的其他文件，在mylayer.py中主要实现3个函数：
  1. `def mylayer_shape(input_shape, ...)`  
  
    |    参数     | 类型 | 说明 |
    | :---------: | :--: | :---------: |
    | input_shape | list | 每个元素代表该层每个输入数据的shape |
    | 其余 | 默认为None | 命名为Caffe模型的model.prototxt中mylayer_param中每个参数的名字 |

     功能：计算出mylayer的输出shape         
     返回：一个list，其中每个元素代表每个输出数据的shape     
     
  2. `def mylayer_layer(inputs, input_shape=None, name=None, ...)` 
  
    |    参数     | 类型 | 说明 |
    | :---------: | :--: | :---------: |
    | inputs | list | 每个元素代表该层每个输入数据 |
    | input_shape | list(默认为None) | 每个元素代表该层每个输入数据的shape |
    | name | str(默认为None) | mylayer的名字 |
    | 其余 | 默认为None | 命名为Caffe模型的model.prototxt中mylayer_param中每个参数的名字 |  
  
     功能：运用PaddlePaddle完成组网来实现`mylayer`的功能         
     返回：一个Variable或Tensor，为组网后的结果
    
  3. `def mylayer_weights(name, data=None)`  
  
    |    参数     | 类型 | 说明 |
    | :---------: | :--: | :---------: |
    | name | str | mylayer的名字 |
    | data | list(默认为None) | 由Caffe模型的model.caffemodel获得的关于mylayer的参数 |
  
     功能：为每个参数（例如kernel、bias等）命名；同时，若Caffe中该层参数与PaddlePaddle中参数的格式不一致，则需要的变换操作也在该函数中实现。     
     返回：一个list，包含每个参数的名字。
     
- 在mylayer.py中注册`mylayer`，主要运用下述代码实现：
  ```
  register(kind='Mylayer', shape=mylayer_shape, layer=mylayer_layer, weights=mylayer_weights)
  # kind为在model.prototxt中mylayer的type
  ```
- 在./x2paddle/op_mapper/caffe_custom_layer/\_\_init\_\_.py中引入该层的使用
  ```
  from . import mylayer
  ```
  
