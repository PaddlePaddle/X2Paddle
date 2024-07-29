import paddle
import numpy as np
np.random.seed(6)
a = np.random.rand(1, 224, 224, 3).astype("float32")
paddle.disable_static()
from pd_model_dygraph_new.x2paddle_code import main
out1=main(a)
print(out1.numpy())
np.save("out1.npy", out1.numpy())

# paddle.disable_static()
# from pd_model_dygraph.x2paddle_code import main
# out2=main(a)
# out1 = np.load("out1.npy")
# print(np.max(np.abs(out1-out2.numpy())))


# import paddle
# import numpy as np
# # ipt = np.load("out1.npy")

# ipt = np.ones([1,3,5,5])
# print(ipt)
# paddle.disable_static()
# data = paddle.to_tensor(data=ipt)
# pool0 = paddle.nn.AvgPool2D(kernel_size=[3, 3], stride=[1, 1], padding='SAME')
# opt0 = pool0(data)
# opt1 = paddle.fluid.layers.pool2d(input=data, pool_size=[3, 3], pool_type='avg', pool_stride=[1, 1], pool_padding='SAME')
# print(np.max(np.abs(opt0.numpy() - opt1.numpy())))
# print(opt0.numpy().shape)
# print(opt1.numpy().shape)