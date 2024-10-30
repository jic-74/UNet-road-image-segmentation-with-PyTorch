import torch
from torch import nn

#### UNet

def doubleConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class EncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.dconv = doubleConv(in_channels, out_channels)
    def forward(self, x):
        x = self.maxpool(x)
        x = self.dconv(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, padding = 0)
        self.dconv = doubleConv(in_channels, out_channels)
        
    def forward(self, x, skip_con):
        x = self.up(x)
        x = torch.cat([x, skip_con], axis = 1)
        x = self.dconv(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inc = doubleConv(in_channels, 64)
        self.down1 = EncoderBlock(64, 128)
        self.down2 = EncoderBlock(128, 256)
        self.down3 = EncoderBlock(256, 512)
        self.down4 = EncoderBlock(512, 1024)
        self.up1 = DecoderBlock(1024, 512)
        self.up2 = DecoderBlock(512, 256)
        self.up3 = DecoderBlock(256, 128)
        self.up4 = DecoderBlock(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x