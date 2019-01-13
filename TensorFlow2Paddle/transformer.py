from paddle_emitter import PaddleEmitter
from tensorflow_parser import TensorflowParser


class Transformer(object):
    def __init__(self, meta_file, ckpt_file, out_nodes, in_shape, in_nodes):
        self.parser = TensorflowParser(meta_file, ckpt_file, out_nodes,
                                       in_shape, in_nodes)
        self.emitter = PaddleEmitter(self.parser.tf_graph)

    def transform_code(self, out_py_file):
        filew = open(out_py_file, 'w')
        codes = self.emitter.gen_code()
        filew.write(codes)
        filew.close()

    def transform_weight(self, out_dir):
        self.emitter.gen_weight(self.parser.weights, out_dir)

    def run(self, dst_dir):
        import os
        if os.path.isdir(dst_dir) or os.path.isfile(dst_dir):
            print("{} already exists, set a new directory")
            return
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        self.transform_code(dst_dir + "/mymodel.py")
        if (len(self.parser.weights) == 0):
            print("There is no tensorflow model weight translate to paddle")
        else:
            self.transform_weight(dst_dir)
