from __future__ import print_function
import tensorflow as tf
import sys

sess = tf.compat.v1.Session()
with open('unet.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    input_map = dict()
    tf.import_graph_def(graph_def, name='', input_map=input_map)

sess.run(tf.compat.v1.global_variables_initializer())

import numpy

numpy.random.seed(13)
data = numpy.random.rand(10, 1, 512, 512).astype('float32')
data = numpy.transpose(data, (0, 2, 3, 1))
numpy.save('input.npy', data)

input = sess.graph.get_tensor_by_name('Placeholder:0')
output = sess.graph.get_tensor_by_name('UNet/conv2d_24/BiasAdd:0')
tmp, = sess.run([output], {input: data})
print(tmp.shape)
numpy.save('result.npy', tmp)
