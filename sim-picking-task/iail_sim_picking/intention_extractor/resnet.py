"""ResNet backbone used by the intention extractor image encoder."""

import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, res_block, layer_list, num_channels=3):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            num_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(res_block, layer_list[0], planes=64, stride=1)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        return self.layer1(x)

    def _make_layer(self, res_block, blocks, planes, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * res_block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * res_block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * res_block.expansion),
            )

        layers.append(
            res_block(
                self.in_channels,
                planes,
                i_downsample=downsample,
                stride=stride,
            )
        )
        self.in_channels = planes * res_block.expansion

        for _ in range(blocks - 1):
            layers.append(res_block(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet50(channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], channels)
