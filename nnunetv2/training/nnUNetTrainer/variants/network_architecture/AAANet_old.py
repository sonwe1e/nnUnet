import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F


class InceptionX(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3, 5]):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels // 4,
            out_channels // 4,
            kernel_size[0],
            padding=kernel_size[0] // 2,
            groups=(
                in_channels // 4 if in_channels < out_channels else out_channels // 4
            ),
        )
        self.conv2 = nn.Conv3d(
            in_channels // 4,
            out_channels // 4,
            kernel_size[1],
            # padding=kernel_size[1] // 2,
            padding=2,
            groups=(
                in_channels // 4 if in_channels < out_channels else out_channels // 4
            ),
            dilation=2,
        )
        self.conv3 = nn.Conv3d(
            in_channels // 2,
            out_channels // 2,
            kernel_size[2],
            padding=kernel_size[2] // 2,
            groups=(
                in_channels // 2 if in_channels < out_channels else out_channels // 2
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
        x1 = x[:, : x.size(1) // 4, :, :, :]
        x2 = x[:, x.size(1) // 4 : x.size(1) // 2, :, :, :]
        x3 = x[:, x.size(1) // 2 :, :, :, :]
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv4(x) + res
        return x


class DefineConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1, 1, 3], expand_rate=2):
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
                            padding=kernel // 2,
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
        self.inception = InceptionX(out_channels, out_channels)

    def forward(self, x_low, x_high):
        x_low = self.up(x_low)
        if self.fusion_mode == "cat":
            x = torch.cat([x_high, self.inception(x_low)], dim=1)
        else:
            # x = x_high + self.inception(x_low)
            x = x_high + x_low
        x = self.upsample(x)
        for extractor in self.extractor:
            x = extractor(x)
        return x


class Out(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # self.conv1 = nn.Conv3d(in_channels, num_classes, 1)

        self.conv1 = nn.Sequential(
            InceptionX(in_channels, in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels, num_classes, 1),
        )

    def forward(self, x):
        p = self.conv1(x)
        return p


class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, patch_size=4):
        super(MultiQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.patchify1 = nn.Conv3d(embed_dim, embed_dim, patch_size, patch_size, 0)
        self.patchify2 = nn.Conv3d(embed_dim, embed_dim, patch_size, patch_size, 0)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, tensor1, tensor2):
        res = tensor2
        _, _, ori_D, ori_H, ori_W = tensor1.size()
        tensor1 = self.patchify1(tensor1)
        tensor2 = self.patchify2(tensor2)
        B, C, D, H, W = tensor1.size()
        # [B, seq_len, embed_dim]
        tensor1_reshaped = tensor1.view(B, C, -1).permute(0, 2, 1)
        tensor1_reshaped = self.layernorm1(tensor1_reshaped)
        tensor2_reshaped = tensor2.view(B, C, -1).permute(0, 2, 1)
        tensor2_reshaped = self.layernorm2(tensor2_reshaped)

        # Linear projections
        Q = self.q_proj(tensor1_reshaped)
        K = self.k_proj(tensor2_reshaped)
        V = self.v_proj(tensor2_reshaped)

        # Reshape for multi-head attention
        B, T, _ = Q.size()
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, num_heads, T, head_dim]
        K = (
            K.view(B, -1, self.head_dim).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        )  # [B, num_heads, T, head_dim]
        V = (
            V.view(B, -1, self.head_dim).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        )  # [B, num_heads, T, head_dim]

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # [B, num_heads, T, T]
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, T, head_dim]
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        )  # [B, T, embed_dim]

        attn_output = self.out_proj(attn_output)

        # 将输出 reshape 回原始形状
        attn_output = attn_output.permute(0, 2, 1).view(B, C, D, H, W)
        attn_output = F.interpolate(
            attn_output,
            size=(ori_D, ori_H, ori_W),
            mode="trilinear",
            align_corners=True,
        )
        attn_output += res

        return attn_output


class MyNet(nn.Module):
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
        self.encoders.append(DefineConv(in_channels, encoder_channels[0]))
        self.aneurysm = nn.ModuleList()
        self.aneurysm.append(DefineConv(in_channels, encoder_channels[0]))
        self.decoders = nn.ModuleList()  # 使用 ModuleList 存储解码器层
        self.mqa_1 = MultiQueryAttention(encoder_channels[-1], 16)
        self.mqa_2 = MultiQueryAttention(encoder_channels[-2], 8)
        self.mqa_3 = MultiQueryAttention(encoder_channels[-3], 8, 2)
        self.mqa_4 = MultiQueryAttention(encoder_channels[-4], 8, 4)

        # 创建编码器层
        for i in range(self.depth):
            self.encoders.append(
                Down(
                    in_channels=encoder_channels[i],
                    out_channels=encoder_channels[i + 1],
                    conv=conv,
                    num_conv=1,
                    stride=strides[i],
                )
            )
        # self.encoders 的权重冻结
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = False

        for i in range(self.depth):
            self.aneurysm.append(
                Down(
                    in_channels=encoder_channels[i],
                    out_channels=encoder_channels[i + 1],
                    conv=conv,
                    num_conv=1,
                    stride=strides[i],
                    kernel_size=[3, 3],
                    expand_rate=1,
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
                    num_conv=0,
                    stride=strides[self.depth - i - 1],
                    fusion_mode="add",
                    kernel_size=[3, 3],
                    expand_rate=1,
                )
            )
        self.out = nn.ModuleList(
            [Out(encoder_channels[depth - i - 1], n_classes) for i in range(depth)]
        )

    def forward(self, x):
        encoder_features = []  # 存储编码器输出
        aneurysm_features = []
        decoder_features = []  # 用于存储解码器特征
        res = x

        # 编码过程
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)

        for aneurysm in self.aneurysm:
            res = aneurysm(res)
            aneurysm_features.append(res)

        # 动脉瘤和血管特征交互
        encoder_features[-1] = self.mqa_1(encoder_features[-1], aneurysm_features[-1])
        encoder_features[-2] = self.mqa_2(encoder_features[-2], aneurysm_features[-2])
        encoder_features[-3] = self.mqa_3(encoder_features[-3], aneurysm_features[-3])
        encoder_features[-4] = self.mqa_4(encoder_features[-4], aneurysm_features[-4])
        # 解码过程
        x_dec = encoder_features[-1]
        for i, decoder in enumerate(self.decoders):
            x_dec = decoder(x_dec, encoder_features[-(i + 2)])
            decoder_features.append(x_dec)  # 保存解码器特征

        if self.deep_supervision:
            return [m(mask) for m, mask in zip(self.out, decoder_features)][::-1]
        else:
            return self.out[-1](decoder_features[-1])
