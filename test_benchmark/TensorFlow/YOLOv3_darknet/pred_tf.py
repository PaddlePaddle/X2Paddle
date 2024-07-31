import tensorflow as tf
import numpy as np
with tf.gfile.GFile('frozen_darknet_yolov3_model.pb', "rb") as pb:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(pb.read())
with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        graph_def,
        name="",  # name可以自定义，修改name之后记得在下面的代码中也要改过来
    )
# for op in graph.get_operations():
#     print(op.name, op.values())
node_in = graph.get_tensor_by_name('inputs:0')  # 此处填入输入节点名称
node_out = graph.get_tensor_by_name('output_boxes:0')  # 此处填入输出节点名称
np.random.seed(6)
image = np.random.rand(1, 416, 416, 3).astype("float32")
np.save("input.npy", image)
with tf.Session(graph=graph) as sess:  # Session()别忘了传入参数！
    # sess.run(tf.global_variables_initializer())  # 因为是从模型中读取，所以无需初始化变量
    feed_dict = {node_in: image}  # image为node_in输入数据，有关代码已省略
    pred = sess.run(node_out, feed_dict)  #运行session，得到node_out
    print(pred)
    np.save("output.npy", pred)
    sess.close()
