from __future__ import print_function
import tensorflow as tf
import sys
import pickle

sess = tf.compat.v1.Session()
with open('yolov3_coco.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    input_map = dict()
    tf.import_graph_def(graph_def, name='', input_map=input_map)

sess.run(tf.compat.v1.global_variables_initializer())

import numpy

data = numpy.load('input.npy')
input = sess.graph.get_tensor_by_name("input/input_data:0")
outputs = list()
outputs.append(sess.graph.get_tensor_by_name("pred_sbbox/concat_2:0"))
outputs.append(sess.graph.get_tensor_by_name("pred_mbbox/concat_2:0"))
outputs.append(sess.graph.get_tensor_by_name("pred_lbbox/concat_2:0"))

result = sess.run(outputs, {input: data})

for r in result:
    print(r.shape)
with open('result.pkl', 'wb') as f:
    pickle.dump(result, f)
