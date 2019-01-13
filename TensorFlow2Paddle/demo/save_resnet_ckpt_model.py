from tensorflow.contrib.slim.nets import resnet_v1 as resnet_v1
import tensorflow.contrib.slim as slim
import tensorflow as tf
import sys


def load_model(ckpt_file):
    img_size = resnet_v1.resnet_v1.default_image_size
    img = tf.placeholder(
        tf.float32, shape=[None, img_size, img_size, 3], name='inputs')
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, endpoint = resnet_v1.resnet_v1_50(
            img, num_classes=None, is_training=False)

    sess = tf.Session()
    load_model = tf.contrib.slim.assign_from_checkpoint_fn(
        ckpt_file, tf.contrib.slim.get_model_variables("resnet_v1_50"))
    load_model(sess)
    return sess


def save_checkpoint(sess, save_dir):
    saver = tf.train.Saver()
    saver.save(sess, save_dir + "/resnet")


if __name__ == "__main__":
    ckpt_file = sys.argv[1]
    save_dir = sys.argv[2]
    sess = load_model(ckpt_file)
    save_checkpoint(sess, save_dir)
