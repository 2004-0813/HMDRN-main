import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel))

    def forward(self, inp):
        return self.layers(inp)

class BackBone(nn.Module):

    def __init__(self, num_channel=64):
        super().__init__()

        self.conv1 = ConvBlock(3, num_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(num_channel, num_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(num_channel, num_channel)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(num_channel, num_channel)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        feature_map_10x10 = x

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        feature_map_5x5 = x

        return feature_map_10x10, feature_map_5x5