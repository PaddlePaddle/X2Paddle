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
import PIL
import numbers
import numpy as np
from PIL import Image
from paddle.vision.transforms import BaseTransform
from paddle.vision.transforms import functional as F


class ToPILImage(BaseTransform):
    def __init__(self, mode=None, keys=None):
        super(ToTensor, self).__init__(keys)
        self.data_format = data_format

    def _apply_image(self, pic):
        """
        Args:
            pic (Tensor|np.ndarray): Image to be converted to PIL Image.
        Returns:
            PIL: Converted image.
        """
        if not (isinstance(pic, paddle.Tensor) or isinstance(pic, np.ndarray)):
            raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(
                type(pic)))

        elif isinstance(pic, paddle.Tensor):
            if pic.ndimension() not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndimension()))

            elif pic.ndimension() == 2:
                # if 2D image, add channel dimension (CHW)
                pic = pic.unsqueeze(0)

        elif isinstance(pic, np.ndarray):
            if pic.ndim not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndim))

            elif pic.ndim == 2:
                # if 2D image, add channel dimension (HWC)
                pic = np.expand_dims(pic, 2)

        npimg = pic
        if isinstance(pic, paddle.Tensor) and "float" in str(pic.numpy(
        ).dtype) and mode != 'F':
            pic = pic.mul(255).byte()
        if isinstance(pic, paddle.Tensor):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))

        if not isinstance(npimg, np.ndarray):
            raise TypeError(
                'Input pic must be a paddle.Tensor or NumPy ndarray, ' +
                'not {}'.format(type(npimg)))

        if npimg.shape[2] == 1:
            expected_mode = None
            npimg = npimg[:, :, 0]
            if npimg.dtype == np.uint8:
                expected_mode = 'L'
            elif npimg.dtype == np.int16:
                expected_mode = 'I;16'
            elif npimg.dtype == np.int32:
                expected_mode = 'I'
            elif npimg.dtype == np.float32:
                expected_mode = 'F'
            if mode is not None and mode != expected_mode:
                raise ValueError(
                    "Incorrect mode ({}) supplied for input type {}. Should be {}"
                    .format(mode, np.dtype, expected_mode))
            mode = expected_mode

        elif npimg.shape[2] == 2:
            permitted_2_channel_modes = ['LA']
            if mode is not None and mode not in permitted_2_channel_modes:
                raise ValueError("Only modes {} are supported for 2D inputs".
                                 format(permitted_2_channel_modes))

            if mode is None and npimg.dtype == np.uint8:
                mode = 'LA'

        elif npimg.shape[2] == 4:
            permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
            if mode is not None and mode not in permitted_4_channel_modes:
                raise ValueError("Only modes {} are supported for 4D inputs".
                                 format(permitted_4_channel_modes))

            if mode is None and npimg.dtype == np.uint8:
                mode = 'RGBA'
        else:
            permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
            if mode is not None and mode not in permitted_3_channel_modes:
                raise ValueError("Only modes {} are supported for 3D inputs".
                                 format(permitted_3_channel_modes))
            if mode is None and npimg.dtype == np.uint8:
                mode = 'RGB'

        if mode is None:
            raise TypeError('Input type {} is not supported'.format(
                npimg.dtype))

        return Image.fromarray(npimg, mode=mode)


class ToTensor(BaseTransform):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to ``numpy.ndarray`` with shapr (C x H x W).
    Args:
        data_format (str, optional): Data format of output tensor, should be 'HWC' or
            'CHW'. Default: 'CHW'.
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.
    """

    def __init__(self, data_format='CHW', keys=None):
        super(ToTensor, self).__init__(keys)
        self.data_format = data_format

    def _apply_image(self, img):
        """
        Args:
            img (PIL.Image|np.ndarray): Image to be converted to tensor.
        Returns:
            np.ndarray: Converted image.
        """
        if isinstance(img, PIL.JpegImagePlugin.JpegImageFile) or isinstance(
                img, PIL.Image.Image):
            img = np.array(img)
        img = img / 255.0
        img = img.transpose((2, 0, 1)).astype("float32")
        img = paddle.to_tensor(img)
        return img


class Normalize(BaseTransform):
    """Normalize the input data with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input data.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (int|float|list): Sequence of means for each channel.
        std (int|float|list): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean=0.0, std=1.0, inplace=False):
        key = None
        super(Normalize, self).__init__(key)
        if isinstance(mean, numbers.Number):
            mean = [mean, mean, mean]

        if isinstance(std, numbers.Number):
            std = [std, std, std]

        self.mean = mean
        self.std = std

    def _apply_image(self, img):
        if isinstance(img, paddle.Tensor):
            img = img.numpy()
        return F.normalize(img, self.mean, self.std, 'CHW', False)


class Lambda(BaseTransform):
    """Apply a user-defined lambda as a transform. This transform does not support torchscript.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError("Argument lambd should be callable, got {}".format(
                repr(type(lambd).__name__)))
        self.lambd = lambd

    def _apply_image(self, img):
        return self.lambd(img)
