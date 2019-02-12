# coding:utf-8
import sys
sys.path.append("..")
from paddle_vgg_19.mymodel import KitModel
import paddle.fluid as fluid
import numpy

use_cuda = True


def model_initialize():
    # 构建模型结构，并初始化参数
    result = KitModel()
    if use_cuda:
        exe = fluid.Executor(fluid.CUDAPlace(0))
    else:
        exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    # 根据save_var.list列表，加载模型参数
    var_list = list()
    global_block = fluid.default_main_program().global_block()
    with open('../paddle_vgg_19/save_var.list') as f:
        for line in f:
            try:
                # 过滤部分不需要加载的参数（OP配置参数）
                var = global_block.var(line.strip())
                var_list.append(var)
            except:
                pass
    fluid.io.load_vars(exe, '../paddle_vgg_19', vars=var_list)

    prog = fluid.default_main_program()
    return exe, prog, result


def test_case(exe, prog, result):
    # 测试随机数据输入
    numpy.random.seed(13)
    img_data = numpy.random.rand(1000, 224, 224, 3)
    # tf中输入为NHWC，PaddlePaddle则为NCHW，需transpose
    img_data = numpy.transpose(img_data, (0, 3, 1, 2))

    # input_0为输入数据的张量名，张量名和数据类型须与my_model.py中定义一致
    for i in range(0, 50):
        r, = exe.run(
            fluid.default_main_program(),
            feed={
                'input_0':
                numpy.array(img_data[i * 20:i * 20 + 20], dtype='float32')
            },
            fetch_list=[result])
        r = r.flatten()
        files = open('fluid_vgg_19.result', 'a+')
        for i in range(0, r.shape[0]):
            files.write(str(r[i]) + '\n')
        files.close()

    # 调用save_inference_model可将模型结构（当前以代码形式保存）和参数均序列化保存
    # 保存后的模型可使用load_inference_model加载
    # http://www.paddlepaddle.org/documentation/docs/zh/1.2/api_cn/api_guides/low_level/inference.html#api-guide-inference
    # fluid.io.save_inference_model("./paddle_model", ["input_0"], [result], exe)


if __name__ == "__main__":
    exe, prog, result = model_initialize()
    test_case(exe, prog, result)
