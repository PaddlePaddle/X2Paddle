#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow.python.framework import graph_util
import tensorflow as tf


def freeze_model(sess, output_tensor_names, freeze_model_path):
    out_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_tensor_names)
    with tf.gfile.GFile(freeze_model_path, 'wb') as f:
        f.write(out_graph.SerializeToString())

    print("freeze model saved in {}".format(freeze_model_path))


import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg
import numpy

with tf.Session() as sess:
    inputs = tf.placeholder(dtype=tf.float32,
                            shape=[None, None, None, 3],
                            name="inputs")
    logits, endpoint = vgg.vgg_16(inputs, num_classes=1000, is_training=False)
    load_model = slim.assign_from_checkpoint_fn(
        "vgg_16.ckpt", slim.get_model_variables("vgg_16"))
    load_model(sess)

    numpy.random.seed(13)
    data = numpy.random.rand(5, 224, 224, 3)
    input_tensor = sess.graph.get_tensor_by_name("inputs:0")
    output_tensor = sess.graph.get_tensor_by_name("vgg_16/fc8/squeezed:0")
    result = sess.run([output_tensor], {input_tensor: data})
    numpy.save("tensorflow.npy", numpy.array(result))

    freeze_model(sess, ["vgg_16/fc8/squeezed"], "vgg16_None.pb")
