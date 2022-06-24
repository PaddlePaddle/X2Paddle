# -*- coding:UTF-8 -*-
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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
import x2paddle
import hashlib
import requests
import threading
import uuid
import json

stats_api = "http://paddlepaddle.org.cn/paddlehub/stat"


def string(param):
    """ 生成字符串。
    """
    return "\'{}\'".format(param)


def check_version():
    version = paddle.__version__
    v0, v1, v2 = version.split('.')
    if not ((v0 == '0' and v1 == '0' and v2 == '0') or
            (int(v0) >= 2 and int(v1) >= 1)):
        return False
    else:
        return True


def _md5(text: str):
    '''Calculate the md5 value of the input text.'''
    md5code = hashlib.md5(text.encode())
    return md5code.hexdigest()


class ConverterCheck(threading.Thread):
    """
    Count the number of calls to model convertion
    """

    def __init__(self,
                 task="ONNX",
                 time_info=None,
                 convert_state=None,
                 lite_state=None,
                 extra_info=None):
        threading.Thread.__init__(self)
        self._task = task
        self._version = x2paddle.__version__
        self._convert_state = convert_state
        self._lite_state = lite_state
        self._extra_info = extra_info
        self._convert_id = _md5(str(uuid.uuid1())[-12:]) + "-" + str(time_info)

    def run(self):
        params = {
            'task': self._task,
            'x2paddle_version': self._version,
            'paddle_version': paddle.__version__,
            'from': 'x2paddle'
        }
        extra = {
            'convert_state': self._convert_state,
            'convert_id': self._convert_id,
        }
        if self._lite_state is not None:
            extra.update({'lite_state': self._lite_state})
        if self._extra_info is not None:
            extra.update(self._extra_info)

        params.update({"extra": json.dumps(extra)})
        try:
            requests.get(stats_api, params, timeout=2)
        except Exception:
            pass

        return


class PaddleDtypes():
    def __init__(self, is_new_version=True):
        if is_new_version:
            self.t_float16 = paddle.float16
            self.t_float32 = paddle.float32
            self.t_float64 = paddle.float64
            self.t_uint8 = paddle.uint8
            self.t_int8 = paddle.int8
            self.t_int16 = paddle.int16
            self.t_int32 = paddle.int32
            self.t_int64 = paddle.int64
            self.t_bool = paddle.bool
        else:
            raise Exception("Paddle>=2.0.0 is required, Please update version!")


is_new_version = check_version()
paddle_dtypes = PaddleDtypes(is_new_version)
