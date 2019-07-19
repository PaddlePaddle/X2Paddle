# X2Paddle
X2Paddle is a toolkit for converting trained model to PaddlePaddle from other deep learning frameworks. 支持主流深度学习框架模型转换至PaddlePaddle（飞桨）


## 转换tensorflow vgg_16模型

### 步骤一 下载模型参数文件
```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
```

### 步骤二 导出vgg_16的pb模型
使用如下python脚本转换
```
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg
from tensorflow.python.framework import graph_util
import tensorflow as tf

def freeze_model(sess, output_tensor_names, freeze_model_path):
    out_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_tensor_names)
    with tf.gfile.GFile(freeze_model_path, 'wb') as f:
        f.write(out_graph.SerializeToString())

    print("freeze model saved in {}".format(freeze_model_path))

with tf.Session() as sess:
    inputs = tf.placeholder(dtype=tf.float32,
                            shape=[None, 224, 224, 3],
                            name="inputs")
    logits, endpoint = vgg.vgg_16(inputs, num_classes=1000, is_training=False)
    load_model = slim.assign_from_checkpoint_fn(
        "vgg_16.ckpt", slim.get_model_variables("vgg_16"))
    load_model(sess)

    freeze_model(sess, ["vgg_16/fc8/squeezed"], "vgg16.pb")
```

### 步骤三 模型转换

```
git clone https://github.com/PaddlePaddle/X2Paddle.git
cd X2Paddle
git checkout develop
export PYTHONPATH=${PWD}
mkdir paddle_model
python x2paddle/convert.py --framework=tensorflow \
                           --model=x2paddle/tests/vgg16.pb \
                           --save_dir=paddle_model
```
