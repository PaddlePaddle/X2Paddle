import paddle
import paddle.nn as nn
from paddle.utils.download import get_weights_path_from_url
from typing import Union, List, Dict, Any, cast
from x2paddle import torch2paddle

__all__ = [
    'VGG',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19_bn',
    'vgg19',
]

model_urls = {
    'vgg11': 'https://x2paddle.bj.bcebos.com/vision/models/vgg11-pt.pdparams',
    'vgg13': 'https://x2paddle.bj.bcebos.com/vision/models/vgg13-pt.pdparams',
    'vgg16': 'https://x2paddle.bj.bcebos.com/vision/models/vgg16-pt.pdparams',
    'vgg19': 'https://x2paddle.bj.bcebos.com/vision/models/vgg19-pt.pdparams',
    'vgg11_bn':
    'https://x2paddle.bj.bcebos.com/vision/models/vgg11_bn-pt.pdparams',
    'vgg13_bn':
    'https://x2paddle.bj.bcebos.com/vision/models/vgg13_bn-pt.pdparams',
    'vgg16_bn':
    'https://x2paddle.bj.bcebos.com/vision/models/vgg16_bn-pt.pdparams',
    'vgg19_bn':
    'https://x2paddle.bj.bcebos.com/vision/models/vgg19_bn-pt.pdparams',
}


class VGG(nn.Layer):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2D((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            torch2paddle.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            torch2paddle.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes), )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2D):
                torch2paddle.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch2paddle.constant_init_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2D):
                torch2paddle.constant_init_(m.weight, 1)
                torch2paddle.constant_init_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch2paddle.normal_init_(m.weight, 0, 0.01)
                torch2paddle.constant_init_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]],
                batch_norm: bool=False) -> nn.Sequential:
    layers: List[nn.Layer] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2D(v), torch2paddle.ReLU(True)]
            else:
                layers += [conv2d, torch2paddle.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512,
        512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
        'M', 512, 512, 512, 512, 'M'
    ],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool,
         **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = get_weights_path_from_url(model_urls[arch])
        model.load_dict(state_dict)
    return model


def vgg11(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, **kwargs)


def vgg11_bn(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, **kwargs)


def vgg13(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, **kwargs)


def vgg13_bn(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, **kwargs)


def vgg16(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, **kwargs)


def vgg16_bn(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, **kwargs)


def vgg19(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, **kwargs)


def vgg19_bn(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, **kwargs)
