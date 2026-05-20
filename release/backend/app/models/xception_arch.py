"""
Xception backbone matching DeepfakeBench checkpoint keys.

Used by both XceptionNet (3-channel input) and F3Net (12-channel input from FAD head).
Key layout:
  backbone.conv1/bn1, backbone.conv2/bn2
  backbone.block1 .. backbone.block12  (entry/middle/exit flow)
  backbone.conv3/bn3 (SeparableConv2d)
  backbone.conv4/bn4 (SeparableConv2d)
  backbone.last_linear  (Linear or Sequential)
  backbone.adjust_channel  (Conv2d + BN, distillation helper — kept for state_dict compatibility)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        reps: int,
        strides: int = 1,
        start_with_relu: bool = True,
        grow_first: bool = True,
    ) -> None:
        super().__init__()
        if out_filters != in_filters or strides != 1:
            self.skip: nn.Module | None = nn.Conv2d(
                in_filters, out_filters, 1, stride=strides, bias=False
            )
            self.skipbn: nn.Module | None = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
            self.skipbn = None

        rep: list[nn.Module] = []
        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, 1, 1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        return x + skip


class Xception(nn.Module):
    """
    DeepfakeBench-style Xception. Accepts arbitrary input channels via `in_channels`
    so the same class supports both XceptionNet (3-ch) and F3Net (12-ch SRM/FAD input).
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3, last_linear_seq: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        if last_linear_seq:
            # F3Net wraps last_linear in a Sequential whose index 1 is the Linear.
            self.last_linear = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(2048, num_classes),
            )
        else:
            self.last_linear = nn.Linear(2048, num_classes)

        # DeepfakeBench distillation helper. Not used during inference but must
        # exist in the module tree for strict state_dict loading to succeed.
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.relu = nn.ReLU(inplace=True)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        # NOTE: post-bn4 ReLU is intentionally omitted (matches DeepfakeBench)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # Global average pool
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
