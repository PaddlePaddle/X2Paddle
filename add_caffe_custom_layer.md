## 如何转换Caffe自定义Layer

本文档介绍如何将Caffe自定义Layer转换为PaddlePaddle模型中的对应实现, 用户可根据自己需要，添加代码实现自定义层，从而支持模型的完整转换。            
***步骤一 下载代码***        
此处涉及修改源码，应先卸载x2paddle，并且下载源码，主要有以下两步完成：
```
pip uninstall x2paddle
pip install git+https://github.com/PaddlePaddle/X2Paddle.git@develop
```

***步骤二 编译caffe.proto***   
该步骤依赖protobuf编译器，其安装过程有以下两种方式：
> 选择一：pip install protobuf          
> 选择二：使用[官方源码](https://github.com/protocolbuffers/protobuf)进行编译

使用脚本./tools/compile.sh将caffe.proto（包含所需的自定义Layer信息）编译成我们所需的目标语言（Python）         
使用方式：
```
bash ./toos/compile.sh /home/root/caffe/src/caffe/proto
# /home/root/caffe/src/caffe/proto为caffe.proto的存放路径，生成的caffe_pb2.py也将保存在该路径下
```

***步骤三 添加自定义Layer的实现代码***
- 进入./x2paddle/op_mapper/caffe_custom_layer，创建.py文件，例如mylayer.py
- 仿照./x2paddle/op_mapper/caffe_custom_layer中的其他文件，在mylayer.py中主要需要实现3个函数，下面以roipooling.py为例分析代码：
  1. `def roipooling_shape(input_shape, pooled_w=None, pooled_h=None)`  
     参数：
     1. input_shape（list）：其中每个元素代表该层每个输入数据的shape，为必须传入的参数   
     2. pooled_w（int）：代表ROI Pooling的kernel的宽，其命名与.prototxt中roi_pooling_param中的key一致
     3. pooled_h（int）：代表ROI Pooling的kernel的高，其命名与.prototxt中roi_pooling_param中的key一致          
     
     功能：计算出进行ROI Pooling后的shape         
     返回：一个list，其中每个元素代表每个输出数据的shape，由于ROI Pooling的输出数据只有一个，所以其list长度为1     
     
  2. `def roipooling_layer(inputs, input_shape=None, name=None, pooled_w=None, pooled_h=None, spatial_scale=None)` 
  
     参数：
     1. inputs（list）：其中每个元素代表该层每个输入数据，为必须传入的参数
     2. input_shape（list）：其中每个元素代表该层每个输入数据的shape，为必须传入的参数  
     3. name（str）：ROI Pooling层的名字，为必须传入的参数  
     4. pooled_w（int）：代表ROI Pooling的kernel的宽，其命名与.prototxt中roi_pooling_param中的key一致
     5. pooled_h（int）：代表ROI Pooling的kernel的高，其命名与.prototxt中roi_pooling_param中的key一致
     6. spatial_scale（float）：用于将ROI坐标从输入比例转换为池化时使用的比例，其命名与.prototxt中roi_pooling_param中的key一致            
     
     功能：运用PaddlePaddle完成组网来实现`roipooling_layer`的功能         
     返回：一个Variable，为组网后的结果
    
  3. `def roipooling_weights(name, data=None)`  

     参数：
     1. name（str）：ROI Pooling层的名字，为必须传入的参数
     2. data（list）：由Caffe模型.caffemodel获得的关于roipooling的参数，roipooling的参数为None
  
     功能：为每个参数（例如kernel、bias等）命名；同时，若Caffe中该层参数与PaddlePaddle中参数的格式不一致，则变换操作也在该函数中实现。     
     返回：一个list，包含每个参数的名字。
     
- 在roipooling.py中注册`roipooling`，主要运用下述代码实现：
  ```
  register(kind='ROIPooling', shape=roipooling_shape, layer=roipooling_layer, weights=roipooling_weights)
  # kind为在model.prototxt中roipooling的type
  ```
- 在./x2paddle/op_mapper/caffe_custom_layer/\_\_init\_\_.py中引入该层的使用
  ```
  from . import roipooling
  ```
  
***步骤三 运行转换代码***
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
