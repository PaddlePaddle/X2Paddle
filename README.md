# X2Paddle
X2Paddle支持将其余深度学习框架训练得到的模型，转换至PaddlePaddle模型。  
X2Paddle is a toolkit for converting trained model to PaddlePaddle from other deep learning frameworks.

## Requirements

python >= 3.5  
paddlepaddle >= 1.5.0  
tensorflow == 1.x  

## Installation
```
pip install git+https://github.com/PaddlePaddle/X2Paddle.git@develop
```

## How To Use
### TensorFlow
```
x2paddle --framework=tensorflow --model=tf_model.pb --save_dir=pd_model
```
### Caffe
```
x2paddle --framework=caffe --proto=deploy.proto --weight=deploy.caffemodel --save_dir=pd_model
```

## Related Docs
1. [X2Paddle使用过程中常见问题](Q&A.md)  
2. [如何导出TensorFlow的pb模型](export_tf_model.md)
