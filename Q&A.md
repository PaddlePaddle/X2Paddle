## 常见问题

**Q1. TensorFlow模型转换过程中，提示『Unknown shape for input tensor[tensor name: "input"]， Please define shape of input here』？**  
A：该提示信息表示无法从TensorFlow的pb模型中获取到输入tensor(tensor名为"input:)的shape信息，所以需要用户手动在提示后输入详细的shape信息，如None,224,224,3 其中None表示Batch 

**Q2. Caffe模型转换过程中，提示『The .py file compiled by .proto file does not work for the old style prototxt. There are 2 solutions for you as below:1. install caffe and don't set '--caffe_proto'.2. modify your .prototxt from the old style to the new style.』？**  
A：该提示信息表示无法使用caffe.proto编译的.py文件来解析老版本的.prototxt，有如下两种方式解决此问题：1.安装caffe并且不设置’--caffe_proto‘；2.将老版本.prototxt修改为新版的.prototxt。
