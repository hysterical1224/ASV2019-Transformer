import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from spp_layer import spatial_pyramid_pool

___author__ = "Shen Song"
__email__ = "1403826619@qq.com"


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size, in_channels=1, sample_rate=16000,
                 stride=32, padding=0, dilation=1, bias=False, groups=1):

        super(SincConv, self).__init__()

        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        # initialize filterbanks using Mel scale
        NFFT = 2048
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)  # Hz to mel conversion
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

        band_pass_filter = self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=3,
                               padding=1,
                               stride=1)

        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               padding=1,
                               kernel_size=3,
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=0,
                                             kernel_size=1,
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)

        self.output_num = [4, 2, 1]

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        # out = spatial_pyramid_pool(x, 1, [int(x.size(-2)), int(x.size(-1))], self.output_num)
        out = self.mp(out)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class Res2NetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=False,  norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = groups * planes
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        self.conv3 = conv1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class Res2Net(nn.Module):
    def __init__(self, d_args, device, zero_init_residual=False,
                 groups=1, width=16, scales=4, se=False, norm_layer=None):
        super(Res2Net, self).__init__()

        self.device = device
        # kernel_size = 1024
        # out_channels = 20
        self.Sinc_conv = SincConv(device=self.device,
                                  out_channels=d_args['filts'],
                                  kernel_size=d_args['first_conv'],
                                  in_channels=d_args['in_channels']
                                  )
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width * scales * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(1, planes[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn0 = norm_layer(1)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Res2NetBottleneck, planes[0], d_args['layers'][0], scales=scales, groups=groups, se=se,
                                       norm_layer=norm_layer)
        self.layer2 = self._make_layer(Res2NetBottleneck, planes[1],  d_args['layers'][1], stride=2, scales=scales, groups=groups,
                                       se=se, norm_layer=norm_layer)
        # self.layer3 = self._make_layer(Res2NetBottleneck, planes[2],  d_args['layers'][2], stride=2, scales=scales, groups=groups,
        #                                se=se, norm_layer=norm_layer)
        # self.layer4 = self._make_layer(Res2NetBottleneck, planes[3],  d_args['layers'][3], stride=2, scales=scales, groups=groups,
        #                                se=se, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[1] * Res2NetBottleneck.expansion,  d_args['num_classes'])
        # self.fc = nn.Linear(planes[3] * Res2NetBottleneck.expansion,  d_args['num_classes'])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Res2NetBottleneck):
                        nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, scales=4, groups=1, se=False, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, downsample, stride=stride, scales=scales, groups=groups, se=se,
                            norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, scales=scales, groups=groups, se=se, norm_layer=norm_layer))

        return nn.Sequential(*layers)




    def forward(self, x):

        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)

        x = self.Sinc_conv(x)
        x = x.unsqueeze(1)
        x = self.bn0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.avg(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x





