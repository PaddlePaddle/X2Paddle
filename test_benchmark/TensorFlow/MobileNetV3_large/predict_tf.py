import paddle
import numpy as np

np.random.seed(6)
a = np.random.rand(1, 224, 224, 3).astype("float32")
paddle.disable_static()
from pd_model_dygraph.x2paddle_code import main

out1 = main(a)
print(out1[0].numpy())
# np.save("out1.npy", out1.numpy())

import tensorflow as tf
import numpy as np

np.random.seed(6)
ipt = np.random.rand(1, 224, 224, 3).astype("float32")
np.save("input.npy", ipt)
with tf.gfile.FastGFile('v3-large_224_1.0_float.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,
                            input_map=None,
                            return_elements=None,
                            name="",
                            op_dict=None,
                            producer_op_list=None)
        with tf.Session() as sess:
            input = graph.get_tensor_by_name('input:0')
            output = graph.get_tensor_by_name(
                'MobilenetV3/Predictions/Softmax:0')
            feed_dict = {input: ipt}
            result = sess.run(output, feed_dict)
            np.save("result.npy", result)
