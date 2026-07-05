"""Vendored GlobalGenerator (pix2pix encoder-decoder) from:
  Microsoft Bringing Old Photos Back to Life (CVPR 2020)
  https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life

Only the generator and ResnetBlock classes are included — discriminator and loss
functions are omitted. Architecture is unchanged from the original source.
"""
from __future__ import annotations

import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        padding_type: str,
        norm_layer,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU(True)
        self.conv_block = self._build(dim, padding_type, norm_layer, activation)

    def _build(self, dim: int, padding_type: str, norm_layer, activation: nn.Module) -> nn.Sequential:
        block: list[nn.Module] = []
        for i in range(2):
            if padding_type == "reflect":
                block.append(nn.ReflectionPad2d(1))
                p = 0
            elif padding_type == "replicate":
                block.append(nn.ReplicationPad2d(1))
                p = 0
            else:
                p = 1
            block.append(nn.Conv2d(dim, dim, kernel_size=3, padding=p))
            block.append(norm_layer(dim))
            if i == 0:
                block.append(activation)
        return nn.Sequential(*block)

    def forward(self, x):
        return x + self.conv_block(x)


class GlobalGenerator(nn.Module):
    """Encoder-decoder with residual bottleneck from BOPBTL's pix2pix pipeline.

    Args:
        input_nc:       Number of input channels (3 for RGB scratch detector).
        output_nc:      Number of output channels (1 for binary scratch mask).
        ngf:            Base filter count.
        n_downsampling: Number of stride-2 downsampling steps before the residual stack.
        n_blocks:       Number of ResNet blocks in the bottleneck.
        norm_layer:     Normalisation layer class (default: BatchNorm2d).
        padding_type:   Padding used inside ResNet blocks.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 1,
        ngf: int = 64,
        n_downsampling: int = 4,
        n_blocks: int = 9,
        norm_layer=None,
        padding_type: str = "reflect",
    ) -> None:
        assert n_blocks >= 0
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        activation = nn.ReLU(True)

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        ]

        for i in range(n_downsampling):
            mult = 2 ** i
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                activation,
            ]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            layers.append(
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, activation=activation)
            )

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            layers += [
                nn.ConvTranspose2d(
                    ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                norm_layer(ngf * mult // 2),
                activation,
            ]

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)