"""A small U-Net built from scratch (no pretrained weights) for 2-class segmentation.

Input : image tensor (B, 3, 256, 256), values 0..1
Output: logits (B, 2, 256, 256) - class 0 = background, class 1 = farmland

Why logits and not probabilities? CrossEntropyLoss expects raw scores.
"""

import torch                   # tensor library
import torch.nn as nn          # layers live here


class ConvBlock(nn.Module):
    """Two 3x3 convolutions + BatchNorm + ReLU. The basic U-Net building block."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),  # 3x3 conv, same HxW
            nn.BatchNorm2d(c_out),        # normalises activations -> more stable training
            nn.ReLU(inplace=True),        # non-linearity
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)          # nn.Sequential calls the layers in order


class UNetSmall(nn.Module):
    """4 down-blocks, a bottleneck, 4 up-blocks, with skip connections."""

    def __init__(self, n_classes: int = 2, base: int = 32):
        super().__init__()
        # encoder: each block halves the resolution (via pooling) and doubles the channels
        self.d1 = ConvBlock(3, base)          # 256x256
        self.d2 = ConvBlock(base, base * 2)   # 128x128
        self.d3 = ConvBlock(base * 2, base * 4)   # 64x64
        self.d4 = ConvBlock(base * 4, base * 8)   # 32x32
        self.pool = nn.MaxPool2d(2)           # halves H and W

        self.bottleneck = ConvBlock(base * 8, base * 16)   # 16x16, deepest point

        # decoder: ConvTranspose2d doubles the resolution back up
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.c4 = ConvBlock(base * 16, base * 8)  # 16 = 8 from up-path + 8 from skip
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.c3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.c2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.c1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, n_classes, kernel_size=1)  # 1x1 conv -> one score per class

    def forward(self, x):
        s1 = self.d1(x)                    # skip 1: 256x256
        s2 = self.d2(self.pool(s1))        # skip 2: 128x128
        s3 = self.d3(self.pool(s2))        # skip 3: 64x64
        s4 = self.d4(self.pool(s3))        # skip 4: 32x32

        b = self.bottleneck(self.pool(s4))  # 16x16

        x = self.u4(b)                              # 32x32
        x = self.c4(torch.cat([x, s4], dim=1))      # concat skip along channel dim
        x = self.u3(x)
        x = self.c3(torch.cat([x, s3], dim=1))
        x = self.u2(x)
        x = self.c2(torch.cat([x, s2], dim=1))
        x = self.u1(x)
        x = self.c1(torch.cat([x, s1], dim=1))
        return self.out(x)                          # (B, 2, 256, 256) logits


if __name__ == "__main__":
    # quick shape + parameter check, runs on CPU
    net = UNetSmall()
    x = torch.randn(2, 3, 256, 256)      # fake batch of 2 images
    print("output:", tuple(net(x).shape))
    print("params:", f"{sum(p.numel() for p in net.parameters()):,}")
