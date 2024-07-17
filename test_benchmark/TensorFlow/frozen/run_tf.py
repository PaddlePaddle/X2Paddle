import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
np.random.seed(6)
ipt0 = np.random.rand(1, 361, 22).astype("float32")
np.random.seed(5)
ipt1 = np.random.rand(1, 19).astype("float32")
np.random.seed(6)
ipt2 = np.random.rand(1, 5).astype("float32")

with tf.Session() as sess:

    with gfile.FastGFile("frozen.pb",'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def,name='')
    sess.run(tf.global_variables_initializer())

    input0 = sess.graph.get_tensor_by_name('swa_model/bin_inputs:0')
    input1 = sess.graph.get_tensor_by_name('swa_model/global_inputs:0')
    input2 = sess.graph.get_tensor_by_name('swa_model/include_history:0')
    out = sess.graph.get_tensor_by_name('swa_model/policy_output:0')
#     out = sess.graph.get_tensor_by_name("swa_model/PadV2:0")
    ret = sess.run(out,  feed_dict={input0: ipt0, input1: ipt1, input2: ipt2})
    print(ret)

import pickle
d = dict()
d["ipt0"] = ipt0
d["ipt1"] = ipt1
d["ipt2"] = ipt2

with open("inputs.pkl", 'wb') as f:
    pickle.dump(d, f)
    
np.save("outputs.npy", ret)