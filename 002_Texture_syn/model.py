import torch.nn as nn
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDown, self).__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, 2, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,
                               3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # inplace = True原地操作
            # nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0, bias=False),
            nn.BatchNorm2d(256)
        )

    def forward(self, x):
        x = x + self.block(x)
        return x
        # return F.relu(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        pre_layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        ]
        mid_layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        ]
        post_layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0, bias=False),
            nn.Tanh()
        ]
        residual_blocks = []
        for _ in range(6):
            residual_blocks.append(ResidualBlock())

        self.pre_conv = nn.Sequential(*pre_layers)
        self.down_conv1 = UNetDown(64, 128)
        self.down_conv2 = UNetDown(128, 256)

        self.residual_conv = nn.Sequential(*residual_blocks)

        self.mid_conv = nn.Sequential(*mid_layers)

        self.up_conv1 = UNetUp(512, 256)
        self.up_conv2 = UNetUp(256, 128)
        self.up_conv3 = UNetUp(128, 64)

        self.post_conv = nn.Sequential(*post_layers)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.down_conv1(x)
        x = self.down_conv2(x)
        x = self.residual_conv(x)
        x = self.mid_conv(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.up_conv3(x)
        x = self.post_conv(x)
        return x


class UNetDown_D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(UNetDown_D, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        pre_layers = [
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        self.pre_conv = nn.Sequential(*pre_layers)
        self.down_conv1 = UNetDown_D(64, 128)
        self.down_conv2 = UNetDown_D(128, 256)
        self.down_conv3 = UNetDown_D(256, 512)
        self.down_conv4 = UNetDown_D(512, 512, 1)
        self.post_conv = nn.Sequential(
            nn.Conv2d(512, 1, 4, stride=1, padding=1))

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.down_conv1(x)
        x = self.down_conv2(x)
        x = self.down_conv3(x)
        x = self.down_conv4(x)
        x = self.post_conv(x)
        return x
