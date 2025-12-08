from __future__ import absolute_import, print_function, division

import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class UDenoiseNet_Narrow(nn.Module):
    # U-net from noise2noise paper/topaz
    def __init__(self, nf=48, top_width=3):
        super(UDenoiseNet_Narrow, self).__init__()
        self.enc1 = nn.Sequential( nn.Conv2d(1, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec5 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf+1, 64, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, top_width, padding=top_width//2)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, x], 1)

        y = self.dec1(h)

        return y


class UDenoiseNet_Width(nn.Module):
    # U-net from noise2noise paper/topaz
    def __init__(self, nf=48, top_width=3):
        super(UDenoiseNet_Width, self).__init__()
        self.enc1 = nn.Sequential( nn.Conv2d(1, nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(nf, nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(nf, nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec5 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 7, padding=3)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf+1, 64, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, top_width, padding=top_width//2)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, x], 1)

        y = self.dec1(h)

        return y

class UDenoiseNet(nn.Module):
    # U-net from noise2noise paper/topaz
    def __init__(self, nf=48, base_width=11, top_width=3):
        super(UDenoiseNet, self).__init__()
        self.enc1 = nn.Sequential( nn.Conv2d(1, nf, base_width, padding=base_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc2 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc3 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc4 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc5 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.MaxPool2d(2)
                                 )
        self.enc6 = nn.Sequential( nn.Conv2d(nf, nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        self.dec5 = nn.Sequential( nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec4 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec3 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2 = nn.Sequential( nn.Conv2d(3*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(2*nf, 2*nf, 3, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1 = nn.Sequential( nn.Conv2d(2*nf+1, 64, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(64, 32, top_width, padding=top_width//2)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv2d(32, 1, top_width, padding=top_width//2)
                                 )

    def forward(self, x):
        # downsampling
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)
        h = self.enc6(p5)

        # upsampling
        n = p4.size(2)
        m = p4.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p4], 1)

        h = self.dec5(h)

        n = p3.size(2)
        m = p3.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p3], 1)

        h = self.dec4(h)

        n = p2.size(2)
        m = p2.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p2], 1)

        h = self.dec3(h)

        n = p1.size(2)
        m = p1.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, p1], 1)

        h = self.dec2(h)

        n = x.size(2)
        m = x.size(3)
        h = F.interpolate(h, size=(n,m), mode='nearest')
        h = torch.cat([h, x], 1)

        y = self.dec1(h)

        return y
