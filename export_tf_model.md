## 如何导出TensorFlow模型

本文档介绍如何将TensorFlow模型导出为X2Paddle支持的模型格式。

TensorFlow提供了接口可将网络参数和网络结构同时保存到同一个文件中，并且只保存指定的前向计算子图，下面示例展示了如何导出tensorflow/models下的VGG16模型

步骤一 下载模型参数文件
```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar xzvf vgg_16_2016_08_28.tar.gz
```

步骤二 加载和导出模型
```
#coding: utf-8
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg
from tensorflow.python.framework import graph_util
import tensorflow as tf

# 固化模型函数
# output_tensor_names: list，指定模型的输出tensor的name
# freeze_model_path: 模型导出的文件路径
def freeze_model(sess, output_tensor_names, freeze_model_path):
    out_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_tensor_names)
    with tf.gfile.GFile(freeze_model_path, 'wb') as f:
        f.write(out_graph.SerializeToString())

    print("freeze model saved in {}".format(freeze_model_path))
    
# 加载模型参数
sess = tf.Session()
inputs = tf.placeholder(dtype=tf.float32,
                        shape=[None, 224, 224, 3],
                        name="inputs")
logits, endpoint = vgg.vgg_16(inputs, num_classes=1000, is_training=False)
load_model = slim.assign_from_checkpoint_fn(
    "vgg_16.ckpt", slim.get_model_variables("vgg_16"))
load_model(sess)

# 导出模型
freeze_model(sess, ["vgg_16/fc8/squeezed"], "vgg16.pb")
```
