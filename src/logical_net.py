#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvINAct(nn.Module):

    def __init__(self, ci, co, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci, co, 3, stride, dilation, dilation=dilation, bias=False), nn.InstanceNorm2d(co, affine=True), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)

class DCSA(nn.Module):

    def __init__(self, c, dilation=2):
        super().__init__()
        hidden = max(c // 8, 8)
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, hidden, 1), nn.ReLU(True), nn.Conv2d(hidden, c, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 3, padding=dilation, dilation=dilation), nn.Sigmoid())
        self.body = nn.Sequential(ConvINAct(c, c), ConvINAct(c, c))

    def forward(self, x):
        y = self.body(x) * self.channel(x)
        s = torch.cat((y.mean(1, keepdim=True), y.amax(1, keepdim=True)), 1)
        return x + y * self.spatial(s)

class LogicALNet(nn.Module):

    def __init__(self, base=48):
        super().__init__()
        self.e1 = nn.Sequential(ConvINAct(3, base), DCSA(base, 1))
        self.e2 = nn.Sequential(ConvINAct(base, base * 2, 2), DCSA(base * 2, 2))
        self.e3 = nn.Sequential(ConvINAct(base * 2, base * 4, 2), DCSA(base * 4, 3))
        self.e4 = nn.Sequential(ConvINAct(base * 4, base * 8, 2), DCSA(base * 8, 4))
        self.u3 = ConvINAct(base * 8 + base * 4, base * 4)
        self.u2 = ConvINAct(base * 4 + base * 2, base * 2)
        self.u1 = ConvINAct(base * 2 + base, base)
        self.recon = nn.Conv2d(base, 5, 1)
        self.seg = nn.Sequential(ConvINAct(base + 5, base), DCSA(base, 2), ConvINAct(base, base // 2), nn.Conv2d(base // 2, 1, 1))

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(a)
        c = self.e3(b)
        d = self.e4(c)
        z = self.u3(torch.cat((F.interpolate(d, size=c.shape[-2:], mode='bilinear', align_corners=False), c), 1))
        z = self.u2(torch.cat((F.interpolate(z, size=b.shape[-2:], mode='bilinear', align_corners=False), b), 1))
        z = self.u1(torch.cat((F.interpolate(z, size=a.shape[-2:], mode='bilinear', align_corners=False), a), 1))
        out = torch.sigmoid(self.recon(z))
        diff = torch.cat((torch.abs(x - out[:, :3]), torch.abs(out[:, 3:4] - x.mean(1, keepdim=True)), out[:, 4:5]), 1)
        return (out[:, :3], out[:, 3:4], out[:, 4:5], self.seg(torch.cat((z, diff), 1)))
