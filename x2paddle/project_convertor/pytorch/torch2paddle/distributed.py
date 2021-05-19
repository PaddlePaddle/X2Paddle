# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import os
import paddle


def init_process_group(backend,
                       init_method=None,
                       timeout=datetime.timedelta(0, 1800),
                       world_size=-1,
                       rank=-1,
                       store=None,
                       group_name=''):
    paddle.distributed.init_parallel_env()
    os.environ['PADDLE_TRAINERS_NUM'] = world_size if world_size > 0 else 1
    os.environ['PADDLE_TRAINER_ID'] = rank if rank > 0 else 1
