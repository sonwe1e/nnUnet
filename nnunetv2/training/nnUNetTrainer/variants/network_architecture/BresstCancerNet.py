import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from torch.amp import autocast
from mamba_ssm import Mamba
from timm.models.layers import DropPath


class EffectiveSEModule(nn.Module):
    def __init__(self, channels, add_maxpool=False):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv3d(channels, channels, kernel_size=1, padding=0)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3, 4), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    @autocast(enabled=False, device_type="cuda")
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out


class SKFusionv2(nn.Module):
    def __init__(self, dim, height=2, reduction=4, kernel_size=3):
        super(SKFusionv2, self).__init__()

        self.height = height
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Conv1d(1, self.height, kernel_size, 1, kernel_size // 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, D, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, D, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.avg_pool(feats_sum)
        attn = attn.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        attn = self.mlp(attn)
        attn = attn.permute(0, 2, 1)
        attn = self.softmax(attn.view(B, self.height, C, 1, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class MSCHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_list=[5, 7, 9]):
        super(MSCHead, self).__init__()
        self.in_channels = in_channels
        self.head1 = nn.Conv3d(in_channels, in_channels // 2, 1, 1, 0)
        self.head2 = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_list[0],
            1,
            kernel_list[0],
            groups=in_channels,
        )
        self.head3 = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_list[1],
            1,
            kernel_list[1],
            groups=in_channels,
        )
        self.head4 = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_list[2],
            1,
            kernel_list[2],
            groups=in_channels,
        )
        self.sk = SKFusionv2(in_channels, height=4, kernel_size=11)
        self.out = nn.Conv3d(in_channels * 5, out_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        x4 = self.head4(x)
        x = self.sk([x1, x2, x3, x4])
        x = torch.cat([x1, x2, x3, x4, x], dim=1)
        x = self.out(x)
        return x


class ConvNeXtConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expand_rate=4,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.conv_list = nn.ModuleList()

        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 7, 1, 3, groups=in_channels),
                nn.InstanceNorm3d(in_channels, affine=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels * expand_rate, 1, 1, 0),
                nn.InstanceNorm3d(in_channels * expand_rate, affine=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels * expand_rate, out_channels, 1, 1, 0),
                nn.InstanceNorm3d(out_channels, affine=True),
            )
        )
        self.residual = in_channels == out_channels
        self.act = nn.LeakyReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)

        if self.drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        res = x
        for _, conv in enumerate(self.conv_list):
            x = conv(x)
            if _ == len(self.conv_list) - 1:
                x += res if self.residual else x
            x = self.act(x)

        if self.drop_path_rate > 0 and self.training:
            x = self.drop_path(x)
        return x


class ResNeXtConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[1, 3, 1],
        expand_rate=2,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.drop_path_rate = drop_path_rate
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

        if self.drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        res = x
        for _, conv in enumerate(self.conv_list):
            x = conv(x)
            if _ == len(self.conv_list) - 1:
                x += res if self.residual else x
            x = self.act(x)

        if self.drop_path_rate > 0 and self.training:
            x = self.drop_path(x)
        return x


class DenseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expand_rate=4,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.conv_list = nn.ModuleList()

        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.InstanceNorm3d(in_channels, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels + in_channels, in_channels * expand_rate, 1, 1, 0
                ),
                nn.InstanceNorm3d(in_channels * expand_rate, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels + in_channels + in_channels * expand_rate,
                    out_channels,
                    1,
                    1,
                    0,
                ),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.conv_list.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels
                    + in_channels
                    + in_channels * expand_rate
                    + out_channels,
                    out_channels,
                    1,
                    1,
                    0,
                ),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.residual = in_channels == out_channels

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.06)
                nn.init.constant_(m.bias, 0)

        if self.drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        res = x
        x1 = self.conv_list[0](x)
        x2 = self.conv_list[1](torch.cat([x, x1], dim=1))
        x3 = self.conv_list[2](torch.cat([x, x1, x2], dim=1))
        x = (
            self.conv_list[3](torch.cat([x, x1, x2, x3], dim=1)) + res
            if self.residual
            else self.conv_list[3](torch.cat([x, x1, x2, x3], dim=1))
        )

        if self.drop_path_rate > 0 and self.training:
            x = self.drop_path(x)
        return x


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_conv=1,
        conv=ResNeXtConv,
        stride=2,
        **kwargs,
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
        conv=ResNeXtConv,
        fusion_mode="add",
        stride=2,
        **kwargs,
    ):
        super().__init__()

        self.fusion_mode = fusion_mode
        self.up = nn.ConvTranspose3d(low_channels, high_channels, stride, stride)
        in_channels = 2 * high_channels if fusion_mode == "cat" else high_channels
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

    def forward(self, x_low, x_high):
        x_low = self.up(x_low)
        x = (
            torch.cat([x_high, x_low], dim=1)
            if self.fusion_mode == "cat"
            else x_low + x_high
        )
        for extractor in self.extractor:
            x = extractor(x)
        return x


class Out(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.4):
        super().__init__()
        self.dp = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv3d(in_channels, num_classes, 1)

    def forward(self, x):
        x = self.dp(x)
        p = self.conv1(x)
        return p


class BresstCancerNet(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        depth=4,
        conv=DenseConv,
        channels=[32, 64, 128, 256, 320],
        encoder_num_conv=[1, 2, 4, 3],
        decoder_num_conv=[1, 2, 4, 3],
        encoder_expand_rate=[1, 2, 2, 3],
        decoder_expand_rate=[1, 2, 2, 3],
        strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)],
        drop_path_rate_list=[0.0, 0.0, 0.0, 0.0],
        deep_supervision=True,
        predict_mode=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.deep_supervision = deep_supervision
        self.predict_mode = predict_mode
        assert len(channels) == depth + 1, "len(encoder_channels) != depth + 1"
        assert len(strides) == depth, "len(strides) != depth"

        self.encoders = nn.ModuleList()  # 使用 ModuleList 存储编码器层
        self.encoders.append(nn.Conv3d(in_channels, channels[0], 3, 1, 1))
        self.decoders = nn.ModuleList()  # 使用 ModuleList 存储解码器层

        # self.positive_feature = nn.Parameter(torch.zeros(1, channels[0], 1, 1, 1))
        # self.negative_feature = nn.Parameter(torch.zeros(1, channels[0], 1, 1, 1))

        # 创建编码器层
        for i in range(self.depth):
            self.encoders.append(
                Down(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    conv=conv,
                    num_conv=encoder_num_conv[i],
                    stride=strides[i],
                    expand_rate=encoder_expand_rate[i],
                    drop_path_rate=drop_path_rate_list[i],
                )
            )

        # 创建解码器层
        for i in range(self.depth):
            self.decoders.append(
                Up(
                    low_channels=channels[self.depth - i],
                    high_channels=channels[self.depth - i - 1],
                    out_channels=channels[self.depth - i - 1],
                    conv=conv,
                    num_conv=decoder_num_conv[self.depth - i - 1],
                    stride=strides[self.depth - i - 1],
                    fusion_mode="add",
                    expand_rate=decoder_expand_rate[self.depth - i - 1],
                    drop_path_rate=0.0,
                )
            )
        self.out = nn.ModuleList(
            [Out(channels[depth - i - 1], n_classes) for i in range(depth)]
        )

    def forward(self, x):
        encoder_features = []  # 存储编码器输出
        decoder_features = []  # 存储解码器输出

        # 编码过程
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)

        # 解码过程
        x_dec = encoder_features[-1]
        for i, decoder in enumerate(self.decoders):
            x_dec = decoder(x_dec, encoder_features[-(i + 2)])
            decoder_features.append(x_dec)  # 保存解码器特征

        # positive_confidence = F.cosine_similarity(
        #     self.positive_feature.expand_as(x_dec), x_dec, dim=1
        # )
        # negative_confidence = F.cosine_similarity(
        #     self.negative_feature.expand_as(x_dec), x_dec, dim=1
        # )
        # confidence = torch.stack([positive_confidence, negative_confidence], dim=1)

        if self.deep_supervision:
            return [m(mask) for m, mask in zip(self.out, decoder_features)][::-1]
        elif self.predict_mode:
            return self.out[-1](decoder_features[-1])
        else:
            return x_dec, self.out[-1](decoder_features[-1])


if __name__ == "__main__":
    model = BresstCancerNet(1, 4)
    summary(model, input_size=(2, 1, 48, 192, 192), device="cuda:5", depth=5)
