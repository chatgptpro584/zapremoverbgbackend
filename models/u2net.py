import torch
import torch.nn as nn

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = nn.functional.interpolate(hx6d, size=hx5.size()[2:], mode='bilinear', align_corners=False)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = nn.functional.interpolate(hx5d, size=hx4.size()[2:], mode='bilinear', align_corners=False)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = nn.functional.interpolate(hx4d, size=hx3.size()[2:], mode='bilinear', align_corners=False)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = nn.functional.interpolate(hx3d, size=hx2.size()[2:], mode='bilinear', align_corners=False)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = nn.functional.interpolate(hx2d, size=hx1.size()[2:], mode='bilinear', align_corners=False)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.outconv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = self.stage1(x)
        d1 = self.outconv(x1)
        return [d1]
