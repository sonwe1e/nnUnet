import torch
import torch.nn as nn
from torchinfo import summary
from torch.cuda.amp import autocast
from einops import rearrange
import torch.nn.functional as F


class InceptionX(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1, 3, 5]):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels // 8,
            out_channels // 8,
            kernel_size[0],
            padding=kernel_size[0] // 2,
            # groups=(
            #     in_channels // 8 if in_channels < out_channels else out_channels // 8
            # ),
        )
        self.conv2 = nn.Conv3d(
            in_channels // 8,
            out_channels // 8,
            kernel_size[1],
            padding=kernel_size[1] // 2,
            groups=(
                in_channels // 8 if in_channels < out_channels else out_channels // 8
            ),
        )
        self.conv3 = nn.Conv3d(
            in_channels // 4,
            out_channels // 4,
            kernel_size[2],
            padding=kernel_size[2] // 2,
            groups=(
                in_channels // 4 if in_channels < out_channels else out_channels // 4
            ),
        )
        self.conv4 = nn.Conv3d(out_channels, out_channels, 1)
        self.act = nn.LeakyReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        x1 = x[:, : x.size(1) // 8, :, :, :]
        x2 = x[:, x.size(1) // 8 : x.size(1) // 4, :, :, :]
        x3 = x[:, x.size(1) // 4 : x.size(1) // 2, :, :, :]
        x4 = x[:, x.size(1) // 2 :, :, :, :]
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv4(x) + res
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // ratio, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class DefineConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[1, 3, 3, 1],
        expand_rate=2,
        se=True,
    ):
        super().__init__()
        expand_rate = expand_rate
        self.conv_list = nn.ModuleList()
        for _, kernel in enumerate(kernel_size):
            if _ == 0:
                self.conv_list.append(
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels,
                            out_channels * expand_rate,
                            kernel,
                            padding=kernel // 2,
                            groups=1,
                        ),
                        nn.InstanceNorm3d(out_channels * expand_rate, affine=True),
                    )
                )
            elif _ == len(kernel_size) - 1:
                self.conv_list.append(
                    nn.Sequential(
                        nn.Conv3d(
                            out_channels * expand_rate,
                            out_channels,
                            kernel,
                            padding=kernel // 2,
                            groups=1,
                        ),
                        nn.InstanceNorm3d(out_channels, affine=True),
                    )
                )
            else:
                self.conv_list.append(
                    nn.Sequential(
                        nn.Conv3d(
                            out_channels * expand_rate,
                            out_channels * expand_rate,
                            kernel,
                            padding=kernel - 1,
                            groups=out_channels * expand_rate,
                            dilation=2,
                        ),
                        nn.InstanceNorm3d(out_channels * expand_rate, affine=True),
                    )
                )

        self.se = SEBlock(out_channels, out_channels) if se else None

        self.residual = in_channels == out_channels
        self.act = nn.LeakyReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        for _, conv in enumerate(self.conv_list):
            x = conv(x)
            if _ == len(self.conv_list) - 1:
                x += res if self.residual else x
            x = self.act(x)
        x = self.se(x) if self.se else x
        return x


class Down(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_conv=1, conv=DefineConv, stride=2, **kwargs
    ):
        super().__init__()
        assert num_conv >= 1, "num_conv must be greater than or equal to 1"
        self.downsample = nn.AvgPool3d(stride)
        self.extractor = nn.ModuleList(
            [
                (
                    conv(in_channels, out_channels, **kwargs)
                    if _ == 0
                    else conv(out_channels, out_channels, **kwargs)
                )
                for _ in range(num_conv)
            ]
        )

    def forward(self, x):
        x = self.downsample(x)
        for extractor in self.extractor:
            x = extractor(x)
        return x


class Up(nn.Module):
    def __init__(
        self,
        low_channels,
        high_channels,
        out_channels,
        num_conv=1,
        conv=DefineConv,
        fusion_mode="cat",
        stride=2,
        **kwargs,
    ):
        super().__init__()

        self.fusion_mode = fusion_mode
        self.up = nn.ConvTranspose3d(low_channels, high_channels, stride, stride)
        self.upsample = (
            conv(2 * high_channels, out_channels, **kwargs)
            if fusion_mode == "cat"
            else conv(high_channels, out_channels, **kwargs)
        )
        self.extractor = nn.ModuleList(
            [conv(out_channels, out_channels, **kwargs) for _ in range(num_conv)]
        )
        self.inceptionx = InceptionX(out_channels, out_channels)

    def forward(self, x_low, x_high):
        x_low = self.up(x_low)
        if self.fusion_mode == "cat":
            x = torch.cat([x_high, x_low], dim=1)
        else:
            x = x_high + self.inceptionx(x_low)
        x = self.upsample(x)
        for extractor in self.extractor:
            x = extractor(x)
        return x


class Out(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels, num_classes, 5, 1, 2),
        )

    def forward(self, x):
        p = self.conv1(x)
        return p


class MyNetM(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        depth=3,
        encoder_channels=[64, 128, 256, 512],
        conv=DefineConv,
        deep_supervision=True,
        strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.deep_supervision = deep_supervision
        assert len(encoder_channels) == depth + 1, "len(encoder_channels) != depth + 1"
        assert len(strides) == depth + 1, "len(strides) != depth"

        self.encoders = nn.ModuleList()  # 使用 ModuleList 存储编码器层
        self.encoders.append(DefineConv(in_channels, encoder_channels[0]))
        # self.encoders.append(nn.Conv3d(in_channels, encoder_channels[0], 3, 1, 1))
        self.decoders = nn.ModuleList()  # 使用 ModuleList 存储解码器层

        # 创建编码器层
        for i in range(self.depth):
            self.encoders.append(
                Down(
                    in_channels=encoder_channels[i],
                    out_channels=encoder_channels[i + 1],
                    conv=conv,
                    num_conv=1,
                    stride=strides[i],
                    se=True,
                )
            )

        # 创建解码器层
        for i in range(self.depth):
            self.decoders.append(
                Up(
                    low_channels=encoder_channels[self.depth - i],
                    high_channels=encoder_channels[self.depth - i - 1],
                    out_channels=encoder_channels[self.depth - i - 1],
                    conv=conv,
                    num_conv=2,
                    stride=strides[self.depth - i - 1],
                    fusion_mode="add",
                    se=True,
                )
            )
        self.out = nn.ModuleList(
            [Out(encoder_channels[depth - i - 1], n_classes) for i in range(depth)]
        )

    def forward(self, x):
        #### Standard
        encoder_features = []  # 存储编码器输出
        decoder_features = []  # 用于存储解码器特征

        # 编码过程
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        # 解码过程
        x_dec = encoder_features[-1]
        for i, decoder in enumerate(self.decoders):
            x_dec = decoder(x_dec, encoder_features[-(i + 2)])
            decoder_features.append(x_dec)  # 保存解码器特征

        out = self.out[-1](decoder_features[-1])
        if self.deep_supervision:
            return [m(mask) for m, mask in zip(self.out, decoder_features)][::-1]
        else:
            return out
