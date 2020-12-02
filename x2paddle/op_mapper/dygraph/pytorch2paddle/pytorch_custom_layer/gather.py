# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.fluid as fluid
from itertools import product
import numpy as np

class Gather(object):
    def __init__(self, dim):
        self.dim = dim
        
    def __call__(self, x, index):
        out_list = list()
        dims = list()
        index_shape = index.shape
        x_type = x.numpy().dtype
        for s in index_shape:
            dims.append(list(range(s))) 
        for id in product(*dims):
            id = list(id)
            id_tensor = paddle.to_tensor(np.array(id).astype('int32'))
            dim_id = paddle.gather_nd(index, id_tensor).numpy()
            id[self.dim] = dim_id
            id_tensor = paddle.to_tensor(np.array(id).astype('int32'))
            data = paddle.gather_nd(x, id_tensor).numpy()
            out_list.append(data)
        out = paddle.to_tensor(np.array(out_list).astype(x_type))
        out = paddle.reshape(out, index_shape)
        return out
