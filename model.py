"""Convolutional autoencoder (with skip connections) for low-light enhancement."""
import torch
import torch.nn as nn


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class AutoEncoder(nn.Module):
    """Encoder compresses features and suppresses noise, decoder rebuilds a brighter image.
    Input and output: RGB tensors in [0, 1]. Height and width must be multiples of 8."""

    def __init__(self):
        super().__init__()
        self.e1 = block(3, 32)
        self.e2 = block(32, 64)
        self.e3 = block(64, 128)
        self.mid = block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = block(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = block(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = block(64, 32)
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))
