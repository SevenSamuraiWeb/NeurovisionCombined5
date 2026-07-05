"""Vendored UNet scratch detector from:
  Microsoft Bringing Old Photos Back to Life (CVPR 2020)
  https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life
  Global/detection_models/networks.py + antialiasing.py

Adapted from the original: `sync_bn` is hard-disabled (no DataParallelWithCallback
wrap), and the antialiased `Downsample` helper is inlined. The `module.` prefix
left on state_dict keys by the original training-time wrap is stripped at load
time in the caller. Architecture is otherwise byte-identical to upstream.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    """Anti-aliased strided pooling from adobe/antialiased-cnns."""

    def __init__(self, pad_type: str = "reflect", filt_size: int = 3, stride: int = 2, channels: int | None = None) -> None:
        super().__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        self.pad_sizes = [
            int((filt_size - 1) / 2),
            int(np.ceil((filt_size - 1) / 2)),
            int((filt_size - 1) / 2),
            int(np.ceil((filt_size - 1) / 2)),
        ]
        coeffs = {
            1: [1.0],
            2: [1.0, 1.0],
            3: [1.0, 2.0, 1.0],
            4: [1.0, 3.0, 3.0, 1.0],
            5: [1.0, 4.0, 6.0, 4.0, 1.0],
            6: [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
            7: [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        }[filt_size]
        a = np.array(coeffs)
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer("filt", filt[None, None, :, :].repeat((channels, 1, 1, 1)))
        if pad_type in ("reflect", "refl"):
            self.pad = nn.ReflectionPad2d(self.pad_sizes)
        elif pad_type in ("replicate", "repl"):
            self.pad = nn.ReplicationPad2d(self.pad_sizes)
        else:
            self.pad = nn.ZeroPad2d(self.pad_sizes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.filt_size == 1:
            return self.pad(inp)[:, :, :: self.stride, :: self.stride]
        return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class UNetConvBlock(nn.Module):
    def __init__(self, conv_num: int, in_size: int, out_size: int, padding: bool, batch_norm: bool) -> None:
        super().__init__()
        block: list[nn.Module] = []
        for _ in range(conv_num):
            block.append(nn.ReflectionPad2d(padding=int(padding)))
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=0))
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            block.append(nn.LeakyReLU(0.2, True))
            in_size = out_size
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, conv_num: int, in_size: int, out_size: int, up_mode: str, padding: bool, batch_norm: bool) -> None:
        super().__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=0),
            )
        self.conv_block = UNetConvBlock(conv_num, in_size, out_size, padding, batch_norm)

    @staticmethod
    def _center_crop(layer: torch.Tensor, target_size: torch.Size) -> torch.Tensor:
        _, _, lh, lw = layer.size()
        dy = (lh - target_size[0]) // 2
        dx = (lw - target_size[1]) // 2
        return layer[:, :, dy : dy + target_size[0], dx : dx + target_size[1]]

    def forward(self, x: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        crop = self._center_crop(bridge, up.shape[2:])
        return self.conv_block(torch.cat([up, crop], 1))


class UNet(nn.Module):
    """BOPBTL detection UNet. Defaults match Global/detection.py exactly."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 4,
        conv_num: int = 2,
        wf: int = 6,
        padding: bool = True,
        batch_norm: bool = True,
        up_mode: str = "upsample",
        with_tanh: bool = False,
        antialiasing: bool = True,
    ) -> None:
        super().__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth - 1

        self.first = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 2 ** wf, kernel_size=7),
            nn.LeakyReLU(0.2, True),
        )
        prev_channels = 2 ** wf

        self.down_path = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(depth):
            if antialiasing:
                self.down_sample.append(
                    nn.Sequential(
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(prev_channels, prev_channels, kernel_size=3, stride=1, padding=0),
                        nn.BatchNorm2d(prev_channels),
                        nn.LeakyReLU(0.2, True),
                        Downsample(channels=prev_channels, stride=2),
                    )
                )
            else:
                self.down_sample.append(
                    nn.Sequential(
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(prev_channels, prev_channels, kernel_size=4, stride=2, padding=0),
                        nn.BatchNorm2d(prev_channels),
                        nn.LeakyReLU(0.2, True),
                    )
                )
            self.down_path.append(
                UNetConvBlock(conv_num, prev_channels, 2 ** (wf + i + 1), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i + 1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(
                UNetUpBlock(conv_num, prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        if with_tanh:
            self.last = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(prev_channels, out_channels, kernel_size=3),
                nn.Tanh(),
            )
        else:
            self.last = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(prev_channels, out_channels, kernel_size=3),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        blocks: list[torch.Tensor] = []
        for i, down_block in enumerate(self.down_path):
            blocks.append(x)
            x = self.down_sample[i](x)
            x = down_block(x)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        return self.last(x)