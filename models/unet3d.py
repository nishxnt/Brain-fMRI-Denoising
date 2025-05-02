import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x): return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x  = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=16):
        super().__init__()
        self.inc        = DoubleConv(in_ch, features)
        self.down1      = Down(features, features*2)
        self.down2      = Down(features*2, features*4)
        self.bottleneck = Down(features*4, features*8)
        self.up2        = Up(features*8, features*4)
        self.up1        = Up(features*4,  features*2)
        self.up0        = Up(features*2,  features)
        self.outc       = nn.Conv3d(features, out_ch, kernel_size=1)

    def forward(self, x):
        # x: [B, T, C, Z, Y, X]
        b, t, c, z, y, x_ = x.shape
        x = x.view(b, t*c, z, y, x_)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.bottleneck(x3)
        x  = self.up2(x4, x3)
        x  = self.up1(x,  x2)
        x  = self.up0(x,  x1)
        return self.outc(x).view(b, t, c, z, y, x_)
