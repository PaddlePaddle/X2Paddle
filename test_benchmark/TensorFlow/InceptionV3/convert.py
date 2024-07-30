from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg
import tensorflow as tf
import numpy

input_checkpoint = "./checkpoint_inceptionv3/rank_model_slim_phaseii0onbase-155999"
save_pb_file = "inceptionv3.pb"


#def freeze_model(sess, output_tensor_names, freeze_model_path):
#def freeze_model(sess, output_tensor_names, var_names_blacklist, freeze_model_path):
def freeze_model(sess, output_tensor_names, var_names_whitelist,
                 freeze_model_path):
    #out_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_tensor_names)
    #     out_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_tensor_names, variable_names_blacklist=var_names_blacklist)
    out_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_tensor_names)

    with tf.gfile.GFile(freeze_model_path, 'wb') as f:
        f.write(out_graph.SerializeToString())

    print("freeze model saved in {}".format(freeze_model_path))


sess = tf.Session()
saver = tf.train.import_meta_graph(input_checkpoint + ".meta",
                                   clear_devices=True)
saver.restore(sess, input_checkpoint)

var_names_whitelist = list()
for v in slim.get_model_variables():
    var_names_whitelist.append(v.name)
#for v in tf.all_variables():
#    print(v, v.name)

var_names_blacklist = list()
for op in tf.get_default_graph().get_operations():
    if not op.type == "Placeholder":
        continue
    var_names_blacklist.append(op.name)

#freeze_model(sess, ["Sigmoid"], save_pb_file)
#freeze_model(sess, ["Sigmoid"], var_names_blacklist, save_pb_file)
freeze_model(sess, ["InceptionV3/Logits/Conv2d_3new_1x1/BiasAdd"],
             var_names_whitelist, save_pb_file)
