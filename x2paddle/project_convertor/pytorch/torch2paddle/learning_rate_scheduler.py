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

import paddle


class ReduceOnPlateau(paddle.optimizer.lr.ReduceOnPlateau):
    def __init__(self,
                 optimizer,
                 mode='min',
                 factor=0.1,
                 patience=10,
                 threshold=0.0001,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 eps=1e-08,
                 verbose=False):
        super().__init__(
            learning_rate=0.01,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            epsilon=eps,
            verbose=verbose)
        optimizer._learning_rate = self


class CosineAnnealingDecay(paddle.optimizer.lr.CosineAnnealingDecay):
    def __init__(self,
                 optimizer,
                 T_max,
                 eta_min=0,
                 last_epoch=-1,
                 verbose=False):
        super().__init__(0.01, T_max, eta_min, last_epoch, verbose)
        optimizer._learning_rate = self


class MultiStepDecay(paddle.optimizer.lr.MultiStepDecay):
    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 verbose=False):
        super().__init__(0.01, milestones, gamma, last_epoch1, verbose)
        optimizer._learning_rate = self
