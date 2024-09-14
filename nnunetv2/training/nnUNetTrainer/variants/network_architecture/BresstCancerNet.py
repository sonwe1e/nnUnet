import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.InstanceNorm3d(in_channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x))) +  x
        return x

class GatingNetwork(nn.Module):
    """Gating Network to select expert weights"""
    def __init__(self, in_features, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(in_features, num_experts)

    def forward(self, x):
        # Assume x has shape (batch_size, in_features)
        gate_logits = self.fc(x)
        gate_weights = F.softmax(gate_logits, dim=1)  # Apply softmax to get expert weights
        return gate_weights


class MoEModule(nn.Module):
    """Mixture of Experts Module"""
    def __init__(self, in_channels, num_experts=4):
        super(MoEModule, self).__init__()
        self.experts = nn.ModuleList([
            BasicConv(in_channels, kernel_size=3, padding=1),
            nn.Sequential(
                BasicConv(in_channels, kernel_size=1, padding=0),
                BasicConv(in_channels, kernel_size=3, padding=1, groups=in_channels),
                BasicConv(in_channels, kernel_size=5, padding=2, groups=in_channels)
            ),
            BasicConv(in_channels, kernel_size=7, padding=3, groups=in_channels),
            nn.Sequential(
                BasicConv(in_channels, kernel_size=5, padding=2, groups=in_channels),
                BasicConv(in_channels, kernel_size=3, padding=1, groups=in_channels),
                BasicConv(in_channels, kernel_size=1, padding=0)
            ),
        ])
        self.gating_network = GatingNetwork(in_channels, num_experts)  # Gating network to control experts

    def forward(self, x):
        # Global average pooling to get a global context for gating network input
        batch_size, in_channels, h, w, d = x.size()
        global_context = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(batch_size, in_channels)  # (batch_size, in_channels)

        # Get gating weights for each expert
        gate_weights = self.gating_network(global_context)  # (batch_size, num_experts)

        # Collect expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch_size, num_experts, out_channels, h, w)

        # Weighted sum of expert outputs
        gate_weights = gate_weights.view(batch_size, len(self.experts), 1, 1, 1, 1)  # (batch_size, num_experts, 1, 1, 1)
        output = torch.sum(expert_outputs * gate_weights, dim=1)  # Weighted sum along the expert dimension

        return output


class SKFusionv2(nn.Module):
    def __init__(self, dim, height=2, reduction=4, kernel_size=3):
        super(SKFusionv2, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

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


class MSCHeadv5(nn.Module):
    def __init__(self, in_channels, out_channels, larger_kenel=11):
        super(MSCHeadv5, self).__init__()
        self.in_channels = in_channels
        self.head1 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.head2 = nn.Conv3d(in_channels, in_channels, (larger_kenel, 1, 1), 1, (larger_kenel // 2, 0, 0))
        self.head3 = nn.Conv3d(in_channels, in_channels, (1, larger_kenel, 1), 1, (0, larger_kenel // 2, 0))
        self.head4 = nn.Conv3d(in_channels, in_channels, (1, 1, larger_kenel), 1, (0, 0, larger_kenel // 2))
        self.head5 = nn.Conv3d(in_channels, in_channels, larger_kenel, 1, larger_kenel // 2, groups=in_channels)
        self.sk = SKFusionv2(in_channels, height=5, kernel_size=9)
        self.out = nn.Conv3d(in_channels * 6, out_channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        x4 = self.head4(x)
        x5 = self.head5(x)
        x = self.sk([x1, x2, x3, x4, x5])
        x = torch.cat([x1, x2, x3, x4, x5, x], dim=1)
        x = self.out(x)
        return x


class ECABlock(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(ECABlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, 1, (kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c, 1)  # (b, c, 1)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.unsqueeze(-1).unsqueeze(-1)


class DefineConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[1, 3, 3, 1],
        expand_rate=2,
        moe=False,
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
        self.moe = MoEModule(out_channels) if moe else nn.Identity()

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
        x = self.moe(x)
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


class BresstCancerNet(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        depth=4,
        encoder_channels=[32, 64, 128, 256, 320],
        conv=DefineConv,
        deep_supervision=False,
        strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.deep_supervision = deep_supervision
        assert len(encoder_channels) == depth + 1, "len(encoder_channels) != depth + 1"
        assert len(strides) == depth, "len(strides) != depth"

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
                    moe=False
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
                    fusion_mode="cat",
                    kernel_size=[1, 3, 3, 1],
                    expand_rate=2,
                    moe=False if i < self.depth - 1 else False
                )
            )
        self.out = nn.ModuleList(
            [Out(encoder_channels[depth - i - 1], n_classes) for i in range(depth)]
        )
        # self.out[-1] = MSCHeadv5(encoder_channels[0], n_classes, 31)

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
    model = BresstCancerNet(1, 4)
    summary(model, input_size=(2, 1, 48, 192, 192), device="cuda:3", depth=5)