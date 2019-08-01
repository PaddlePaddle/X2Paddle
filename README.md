# X2Paddle
X2Paddle支持将其余深度学习框架训练得到的模型，转换至PaddlePaddle模型。  
X2Paddle is a toolkit for converting trained model to PaddlePaddle from other deep learning frameworks.

## 环境依赖

python >= 3.5  
paddlepaddle >= 1.5.0  
tensorflow == 1.x  

## 安装
```
pip install git+https://github.com/PaddlePaddle/X2Paddle.git@develop
```

## 使用方法
### TensorFlow
```
x2paddle --framework=tensorflow --model=tf_model.pb --save_dir=pd_model
```
### Caffe
```
x2paddle --framework=caffe --proto=deploy.proto --weight=deploy.caffemodel --save_dir=pd_model
```
### 参数选项
| 参数 | |
|----------|--------------|
|--framework | 源模型类型 (tensorflow、caffe) |
|--proto | 当framework为caffe时，该参数指定caffe模型的proto文件路径 |
|--weight | 当framework为caffe时，该参数指定caffe模型的参数文件路径 |
|--save_dir | 指定转换后的模型保存目录路径 |
|--model | 当framework为tensorflow时，该参数指定tensorflow的pb模型文件路径 |

## Related Docs
1. [X2Paddle使用过程中常见问题](Q&A.md)  
2. [如何导出TensorFlow的pb模型](export_tf_model.md)