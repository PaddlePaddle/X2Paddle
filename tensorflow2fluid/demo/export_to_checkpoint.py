from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import vgg as vgg
from tensorflow.contrib.slim.nets import resnet_v1 as resnet_v1
from tensorflow.contrib.framework.python.ops import arg_scope
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy
from six import text_type as _text_type

def inception_v3(ckpt_file):
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

def resnet_v1_50(ckpt_file):
    img_size = resnet_v1.resnet_v1.default_image_size
    img = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='inputs')
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, endpoint = resnet_v1.resnet_v1_50(img, num_classes=1000, is_training=False)

    sess = tf.Session()
    load_model = tf.contrib.slim.assign_from_checkpoint_fn(ckpt_file, tf.contrib.slim.get_model_variables("resnet_v1_50"))
    load_model(sess)
    return sess

def resnet_v1_101(ckpt_file):
    img_size = resnet_v1.resnet_v1.default_image_size
    img = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='inputs')
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, endpoint = resnet_v1.resnet_v1_101(img, num_classes=1000, is_training=False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    load_model = tf.contrib.slim.assign_from_checkpoint_fn(ckpt_file, tf.contrib.slim.get_model_variables("resnet_v1_101"))
    load_model(sess)
    return sess

def vgg_16(ckpt_file):
    img_size = vgg.vgg_16.default_image_size
    inputs = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3],
            name="inputs")
    logits, endpoint = vgg.vgg_16(inputs, num_classes=1000, is_training=False)
    sess = tf.Session()

    load_model = tf.contrib.slim.assign_from_checkpoint_fn(ckpt_file,
            tf.contrib.slim.get_model_variables("vgg_16"))
    load_model(sess)
    return sess

def vgg_19(ckpt_file):
    img_size = vgg.vgg_19.default_image_size
    inputs = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3],
            name="inputs")
    logits, endpoint = vgg.vgg_19(inputs, num_classes=1000, is_training=False)
    sess = tf.Session()
    load_model = tf.contrib.slim.assign_from_checkpoint_fn(ckpt_file,
            tf.contrib.slim.get_model_variables("vgg_19"))
    load_model(sess)
    return sess

def save_checkpoint(sess, save_dir):
    saver = tf.train.Saver()
    saver.save(sess, save_dir+"/model")

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=_text_type, default=None, help="inception_v3/resnet_v1_50/resnet_v1_101/vgg_16/vgg_19")
    parser.add_argument("--ckpt_file", "-c", type=_text_type, default=None, help="parameters ckpt file")
    parser.add_argument("--save_dir", "-s", type=_text_type, default=None, help="model path")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    sess = None
    if args.model is None or args.save_dir is None or args.ckpt_file is None:
        raise Exception("--model, --ckpt_file and --save_dir are needed")
    if args.model == "inception_v3":
        sess = inception_v3(args.ckpt_file)
    elif args.model == "resnet_v1_50":
        sess = resnet_v1_50(args.ckpt_file)
    elif args.model == "resnet_v1_101":
        sess = resnet_v1_101(args.ckpt_file)
    elif args.model == "vgg_16":
        sess = vgg_16(args.ckpt_file)
    elif args.model == "vgg_19":
        sess = vgg_19(args.ckpt_file)
    else:
        raise Exception("Only support inception_v3/resnet_v1_50/resnet_v1_101/vgg_16/vgg_19")
    save_checkpoint(sess, args.save_dir)
