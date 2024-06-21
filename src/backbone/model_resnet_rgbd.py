import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, ReLU, Dropout, MaxPool2d, Sequential, Module


# Support: ['ResNet_50', 'ResNet_101', 'ResNet_152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""

    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""

    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetRGBD(Module):

    def __init__(self, input_size, block, layers, embedding_size, zero_init_residual=True):
        super(ResNetRGBD, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"

        self.inplanes = 64

        # RGB branch
        self.rgb_conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.rgb_bn1 = BatchNorm2d(64)
        self.rgb_relu = ReLU(inplace=True)
        self.rgb_maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rgb_layer1 = self._make_layer(block, 64, layers[0])
        self.rgb_layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.rgb_layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.rgb_layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Depth branch
        self.inplanes = 64
        self.depth_conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_bn1 = BatchNorm2d(64)
        self.depth_relu = ReLU(inplace=True)
        self.depth_maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.depth_layer1 = self._make_layer(block, 64, layers[0])
        self.depth_layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.depth_layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.depth_layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn_o1 = BatchNorm2d(4096)
        self.dropout = Dropout()
        if input_size[0] == 112:
            self.fc = Linear(4096 * 4 * 4, embedding_size)
        else:
            self.fc = Linear(4096 * 8 * 8, embedding_size)
        self.bn_o2 = BatchNorm1d(embedding_size)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):

        # Assuming x is the input tensor with shape [batch_size, 4, 244, 244]
        # Split the 4-channel input into 3-channel RGB and 1-channel depth images
        rgb_x = x[:, :3, :, :]  # Take the first 3 channels for RGB
        depth_x = x[:, 3:4, :, :]  # Take the fourth channel for depth
        depth_x = torch.cat((depth_x, depth_x, depth_x), dim=1)

        # RGB branch
        rgb_x = self.rgb_conv1(rgb_x)
        rgb_x = self.rgb_bn1(rgb_x)
        rgb_x = self.rgb_relu(rgb_x)
        rgb_x = self.rgb_maxpool(rgb_x)
        rgb_x = self.rgb_layer1(rgb_x)
        rgb_x = self.rgb_layer2(rgb_x)
        rgb_x = self.rgb_layer3(rgb_x)
        rgb_x = self.rgb_layer4(rgb_x)

        # Depth branch
        depth_x = self.depth_conv1(depth_x)
        depth_x = self.depth_bn1(depth_x)
        depth_x = self.depth_relu(depth_x)
        depth_x = self.depth_maxpool(depth_x)
        depth_x = self.depth_layer1(depth_x)
        depth_x = self.depth_layer2(depth_x)
        depth_x = self.depth_layer3(depth_x)
        depth_x = self.depth_layer4(depth_x)

        # Fusion
        fusion_x = torch.cat((rgb_x, depth_x), 1)

        fusion_x = self.bn_o1(fusion_x)
        fusion_x = self.dropout(fusion_x)
        fusion_x = fusion_x.view(fusion_x.size(0), -1)
        fusion_x = self.fc(fusion_x)
        fusion_x = self.bn_o2(fusion_x)

        return fusion_x


def ResNet_50_rgbd(input_size, embedding_size, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNetRGBD(input_size, Bottleneck, [3, 4, 6, 3], embedding_size, **kwargs)

    return model


def ResNet_101_rgbd(input_size, embedding_size, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNetRGBD(input_size, Bottleneck, [3, 4, 23, 3], embedding_size, **kwargs)

    return model


def ResNet_152_rgbd(input_size, embedding_size, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNetRGBD(input_size, Bottleneck, [3, 8, 36, 3], embedding_size, **kwargs)

    return model
