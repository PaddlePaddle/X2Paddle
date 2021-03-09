# # 多重继承
# class One:
#     def __init__(self):
#         print(1)
        
# class Two:
#     def __init__(self, num=0):
#         print(num)
        
# class C(One, Two):
#     def __init__(self, num=1):
#         if num == 1:
#             One.__init__(self)
#         else:
#             Two.__init__(self,num=num)
# c = C(num=3)


'''
import torch
import numpy as np
a = torch.tensor(np.array([[[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]]).astype("float32"))
mask = torch.tensor(np.array([True, False]).astype("bool"))
a = a[mask, ...]
a = torch.tensor(np.array([1,2,3]).astype("float32"))
mask = torch.tensor(np.array([True, False, True]).astype("bool"))
a = a[mask]
print(a)
import paddle
import numpy as np
# from x2paddle import torch2paddle
# a = paddle.to_tensor(np.array([[[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]]).astype("float32"))
# mask = paddle.to_tensor(np.array([True, False]).astype("bool"))
# a = a[mask, ...]
a = paddle.to_tensor(np.array([1,2,3]).astype("float32"))
mask = paddle.to_tensor(np.array([True, False, True]).astype("bool"))
a = a[mask]
print(a)
'''

'''
import torch
import numpy as np
np.random.seed(6)
a = torch.tensor(np.random.rand(1, 2, 3).astype("float32"))
print(a[..., 2])
import paddle
import numpy as np
from x2paddle import torch2paddle
np.random.seed(6)
a = paddle.to_tensor(np.random.rand(1, 2, 3).astype("float32"))
print(a[..., 2])
'''

import paddle
import numpy as np
from x2paddle import torch2paddle
a = paddle.to_tensor(np.array([1,2,3]).astype("float32"))
mask = paddle.to_tensor(np.array([True, False, True]).astype("bool"))
a[mask] = 0
print(a)