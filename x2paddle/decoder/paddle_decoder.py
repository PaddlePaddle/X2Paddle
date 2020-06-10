import paddle.fluid as fluid


class PaddleDecoder(object):
    def __init__(self,
                 model_dir,
                 model_filename='__model__',
                 params_filename=None):
        exe = fluid.Executor(fluid.CPUPlace())
        [self.program, feed, fetchs] = fluid.io.load_inference_model(
            model_dir,
            exe,
            model_filename=model_filename,
            params_filename=params_filename)
