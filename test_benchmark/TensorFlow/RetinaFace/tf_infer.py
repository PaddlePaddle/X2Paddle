from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy
import sys
import cv2

sess = tf.Session()
with gfile.FastGFile('frozen_eval_graph_v1_500000.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())

input0 = sess.graph.get_tensor_by_name('input:0')
import numpy
numpy.random.seed(13)
data0 = numpy.random.rand(2, 3, 640, 640).astype('float32')
data0 = numpy.transpose(data0, (0, 2, 3, 1))
numpy.save('input.npy', data0)
outputs = [None] * 12
outputs[0] = sess.graph.get_tensor_by_name("rf/det_1/concat:0")
outputs[1] = sess.graph.get_tensor_by_name("rf/det_2/concat:0")
outputs[2] = sess.graph.get_tensor_by_name("rf/det_3/concat:0")
outputs[3] = sess.graph.get_tensor_by_name("rf/oup_1/score:0")
outputs[4] = sess.graph.get_tensor_by_name("rf/oup_1/conv_2/Conv2D:0")
outputs[5] = sess.graph.get_tensor_by_name("rf/oup_1/conv_3/Conv2D:0")
outputs[6] = sess.graph.get_tensor_by_name("rf/oup_2/score:0")
outputs[7] = sess.graph.get_tensor_by_name("rf/oup_2/conv_2/Conv2D:0")
outputs[8] = sess.graph.get_tensor_by_name("rf/oup_2/conv_3/Conv2D:0")
outputs[9] = sess.graph.get_tensor_by_name("rf/oup_3/score:0")
outputs[10] = sess.graph.get_tensor_by_name("rf/oup_3/conv_2/Conv2D:0")
outputs[11] = sess.graph.get_tensor_by_name("rf/oup_3/conv_3/Conv2D:0")
result = sess.run(outputs, {input0:data0})

for r in result:
    print(r.shape)

import pickle
with open('result.pkl', 'wb') as f:
    pickle.dump(result, f)
