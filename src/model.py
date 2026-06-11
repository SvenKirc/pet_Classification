import torch
from torch import nn

class ConvolutionalPet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            self.conv_block(3, 16),
            self.conv_block(16, 32),
            self.conv_block(32, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 37)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x