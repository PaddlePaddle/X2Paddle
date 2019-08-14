#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import functools


def walk_dir(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            yield file


def calc_diff(f1, f2):
    import numpy as np

    d1 = np.load(f1)
    d2 = np.load(f2)

    d1 = d1.flatten()
    d2 = d2.flatten()

    d1_num = functools.reduce(lambda x, y: x * y, d1.shape)
    d2_num = functools.reduce(lambda x, y: x * y, d2.shape)
    if d1_num != d2_num:
        print(d1.shape)
        print(d2.shape)
        assert (d1_num == d2_num), "their shape is not consistent"

    try:
        mask = np.abs(d1) >= np.abs(d2)
        mask = mask.astype('int32')
        df = np.abs(d1 - d2)
        max_df = np.max(df)
        df = df / (1.0e-10 + np.abs(d1) * mask + np.abs(d2) * (1 - mask))
        sq_df = np.mean(df * df)
        return max_df, sq_df
    except Exception as e:
        return 1.0, 1.0


def compare(path1, path2, no_exception):
    def diff(f1, f2):
        max_df, sq_df = calc_diff(f1, f2)
        print('[max_df:%.4e, sq_df:%.4e] when compare %s <=> %s' %
              (max_df, sq_df, os.path.basename(f1), os.path.basename(f2)))
        if no_exception is False:
            assert (max_df < 1e-4) or (sq_df < 1e-4), \
                    'diff is too large with value[%.6e]' % (max_df)

    if os.path.exists(path1) is False:
        print('not found %s' % (path1))
        return 1
    elif os.path.exists(path2) is False:
        print('not found %s' % (path2))
        return 1

    if path1.find('.npy') > 0 and path2.find('.npy') > 0:
        diff(path1, path2)
        return

    for f in walk_dir(path2):
        if f.find('.npy') < 0:
            continue

        f1 = os.path.join(path1, f)
        f2 = os.path.join(path2, f)
        if os.path.exists(f1) and os.path.exists(f2):
            diff(f1, f2)

    print('all checking succeed to pass')
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        path1 = 'lenet.tf/results'
        path2 = 'lenet.paddle/results'
    elif len(sys.argv) >= 3:
        path1 = sys.argv[1]
        path2 = sys.argv[2]
        if len(sys.argv) == 4:
            no_exception = True
        else:
            no_exception = False
    else:
        print('usage:')
        print(' %s [path1] [path2]' % (sys.argv[0]))
        exit(1)

    exit(compare(path1, path2, no_exception))
