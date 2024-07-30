import torch
import torch.nn as nn
from math import sqrt
#from torchstat import stat as tstat


class SeparableConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   1,
                                   1,
                                   0,
                                   1,
                                   1,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, base_channels):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=base_channels,
                               out_channels=base_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False,
                               padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=base_channels,
                               out_channels=base_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False,
                               padding_mode='reflect')

        self.norm1 = nn.InstanceNorm2d(base_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(base_channels, affine=True)

    def forward(self, x):
        res = self.norm1(x)
        res = self.relu(res)
        res = self.conv1(res)
        res = self.norm2(res)
        res = self.relu(res)
        res = self.conv2(res)

        res1 = res + x
        return res1


class Net(nn.Module):

    def __init__(self, base_channels, depth_num):
        super(Net, self).__init__()
        self.input = nn.Conv2d(in_channels=3,
                               out_channels=base_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False,
                               padding_mode='reflect')
        self.conv1 = nn.Conv2d(in_channels=base_channels,
                               out_channels=2 * base_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False,
                               padding_mode='reflect')
        self.residual_layer = self.make_layer(ResBlock, depth_num,
                                              base_channels * 2)
        self.conv2 = nn.ConvTranspose2d(in_channels=2 * base_channels,
                                        out_channels=base_channels,
                                        kernel_size=2,
                                        stride=2,
                                        padding=0,
                                        bias=False)
        self.output = nn.Conv2d(in_channels=base_channels,
                                out_channels=3,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                                padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        res = self.relu(self.input(x))
        res = self.conv1(res)
        res = self.residual_layer(res)
        #本来没有，后来添加的
        res = self.relu(res)

        res = self.conv2(res)
        res = self.output(res)
        res = res + x
        return res


if __name__ == "__main__":
    model = Net(5, 4)
    #tstat(model, (3, 640, 360))
    t = torch.zeros((1, 3, 640, 360))
    print(model(t).shape)
    print("!!!")
