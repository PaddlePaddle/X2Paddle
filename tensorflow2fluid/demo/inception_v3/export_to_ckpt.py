from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.framework.python.ops import arg_scope
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy
import sys
numpy.random.seed(13)

ckpt_file = sys.argv[1]
checkpoint_dir = sys.argv[2]

def get_tuned_variables():
    CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []

    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

def load_model():
    img_size = inception.inception_v3.default_image_size
    img = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='inputs')
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(img, num_classes=1000, is_training=False)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    load_model = tf.contrib.slim.assign_from_checkpoint_fn(ckpt_file, get_tuned_variables(), ignore_missing_vars=True)
    load_model(sess)
    return sess

def save_checkpoint(sess):
    saver = tf.train.Saver()
    saver.save(sess, checkpoint_dir+"/model")

if __name__ == "__main__":
    sess = load_model()
    save_checkpoint(sess)
