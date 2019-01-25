from paddle_emitter import PaddleEmitter
from tensorflow_parser import TensorflowCkptParser
from tensorflow_parser import TensorflowPbParser


class Transformer(object):
    def __init__(self, meta_file, ckpt_file, out_nodes, in_shape, in_nodes, save_dir):
        self.parser = TensorflowCkptParser(meta_file, ckpt_file, out_nodes,
                                       in_shape, in_nodes)
        self.emitter = PaddleEmitter(self.parser, save_dir)

    def transform_code(self):
        codes = self.emitter.run()

    def run(self):
        self.transform_code()

class PbTransformer(object):
    def __init__(self, pb_file, out_nodes, in_shape, in_nodes, save_dir):
        self.parser = TensorflowPbParser(pb_file, out_nodes, in_shape, in_nodes)
        self.emitter = PaddleEmitter(self.parser, save_dir)
        node = self.parser.tf_graph.tf_graph.node[0]

    def transform_code(self):
        codes = self.emitter.run()

    def run(self):
        self.transform_code()

