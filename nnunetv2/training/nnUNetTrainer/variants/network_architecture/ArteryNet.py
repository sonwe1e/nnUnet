import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F


class DefineConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=[1, 3, 3, 1], expand_rate=2
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
                            padding=kernel // 2,
                            groups=out_channels * expand_rate,
                        ),
                        nn.InstanceNorm3d(out_channels * expand_rate, affine=True),
                    )
                )
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
        fusion_mode="add",
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

    def forward(self, x_low, x_high):
        x_low = self.up(x_low)
        if self.fusion_mode == "cat":
            x = torch.cat([x_high, x_low], dim=1)
        else:
            x = x_high + x_low
        x = self.upsample(x)
        for extractor in self.extractor:
            x = extractor(x)
        return x


class Out(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, num_classes, 1)

    def forward(self, x):
        p = self.conv1(x)
        return p


class ArteryNet(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        depth=4,
        encoder_channels=[32, 64, 128, 256, 320],
        conv=DefineConv,
        deep_supervision=False,
        strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.deep_supervision = deep_supervision
        assert len(encoder_channels) == depth + 1, "len(encoder_channels) != depth + 1"
        assert len(strides) == depth + 1, "len(strides) != depth"

        self.encoders = nn.ModuleList()  # 使用 ModuleList 存储编码器层
        self.encoders.append(nn.Conv3d(in_channels, encoder_channels[0], 3, 1, 1))
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
                    kernel_size=[1, 3, 3, 1],
                    expand_rate=2,
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
                    num_conv=1,
                    stride=strides[self.depth - i - 1],
                    fusion_mode="add",
                    kernel_size=[1, 3, 3, 1],
                    expand_rate=2,
                )
            )
        self.out = nn.ModuleList(
            [Out(encoder_channels[depth - i - 1], n_classes) for i in range(depth)]
        )

    def forward(self, x):
        encoder_features = []  # 存储编码器输出
        decoder_features = []  # 存储解码器输出
        res = x

        # 编码过程
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)

        # 解码过程
        x_dec = encoder_features[-1]
        for i, decoder in enumerate(self.decoders):
            x_dec = decoder(x_dec, encoder_features[-(i + 2)])
            decoder_features.append(x_dec)  # 保存解码器特征

        if self.deep_supervision:
            return [m(mask) for m, mask in zip(self.out, decoder_features)][::-1]
        else:
            return self.out[-1](decoder_features[-1])


if __name__ == "__main__":
    model = ArteryNet(1, 4)
    print(summary(model, input_size=(1, 64, 64, 64), device="cpu"))
