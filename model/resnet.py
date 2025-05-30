"""Imported from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
and added support for the 1x32x32 mel spectrogram for the speech recognition.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385
"""

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    # Định nghĩa lớp tích chập kernel 3x3
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    # Khối basicblock có 2 tầng tích chập, sử dụng cộng phần dư residual block
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    # Forward các bước BasicBlock được trình bày trong Báo Cáo
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual 
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, in_channels=1):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=7, stride=1, padding=3,
                               bias=False)  # Lớp tích chập kernel_size 7x7, padding = 3 để giữ kích thước, tăng số kênh từ 1 lên 16
        self.bn1 = nn.BatchNorm2d(16)       # Lớp BatchNorm cho tensor 4 chiều, 16 kênh
        self.relu = nn.ReLU(inplace=True)   # Hàm kích hoạt ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # Lấy giá trị lớn nhất trên không gian 2d, kernel lấy mẫu 3x3, stride = 2, padding = 1
        self.layer1 = self._make_layer(block, 16, layers[0])                # Định nghĩa lớp make_layer 1 bao gồm 2 lớp BasicBlock
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)      # Định nghĩa lớp make_layer 2 bao gồm 2 lớp BasicBlock, stride = 2 để giảm chiều dữ liệu
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)      # Định nghĩa lớp make_layer 3 bao gồm 2 lớp BasicBlock, stride = 2 để giảm chiều dữ liệu
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)     # Định nghĩa lớp make_layer 4 bao gồm 2 lớp BasicBlock, stride = 2 để giảm chiều dữ liệu
        self.avgpool = nn.AvgPool2d(1, stride=1)                            # Giống Maxpool nhưng lấy trung bình thay vì lấy max
        self.fc = nn.Linear(128 * block.expansion, num_classes)             # Lớp fc sử dụng tầng trọng số tuyến tính đầu vào (1, 128) đầu ra (1, n_classes)

        # Khởi tạo các tham số mô hình batchNorm và các tầng trọng số, tích chập
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        # Định nghĩa lớp make_layer
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    # Forward lớp ResNet, không sử dụng
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    # Định nghĩa lớp ResNet18 
def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model
