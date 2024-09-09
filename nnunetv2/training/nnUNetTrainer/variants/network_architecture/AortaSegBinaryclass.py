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
            groups=(
                in_channels // 8 if in_channels < out_channels else out_channels // 8
            ),
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


class ViTLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        # print(f"ViTLayer: dim: {dim}")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = rearrange(x, "b c d h w -> b c (d h w)")

        x = x.permute(2, 0, 1)  # rearrange to [seq_len, batch, dim]
        x1 = self.norm1(x)
        x2, _ = self.attn(x1, x1, x1)
        x = x + x2
        x2 = self.mlp(self.norm2(x))
        x = x + x2
        x = x.permute(1, 2, 0)  # rearrange back to [batch, seq_len, dim]
        x = rearrange(x, "b c (d h w) -> b c d h w", d=D, h=H, w=W)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        # print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.channel_token = channel_token  ## whether to use channel as tokens

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert (
            x_flat.shape[2] == d_model
        ), f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
        return out

    @autocast(enabled=True)
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x) + x
        else:
            out = self.forward_patch_token(x) + x

        return out


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
        expand_rate=1,
        se=True,
        mamba=False,
        vit=False,
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
                            padding=kernel // 2,
                            groups=out_channels * expand_rate,
                        ),
                        nn.InstanceNorm3d(out_channels * expand_rate, affine=True),
                    )
                )

        self.se = SEBlock(out_channels, out_channels) if se else None
        self.mamba = (
            MambaLayer(dim=out_channels, d_state=16, d_conv=4, expand=2)
            if mamba
            else None
        )
        self.vit = (
            ViTLayer(dim=out_channels, heads=16, mlp_dim=2 * out_channels)
            if vit
            else None
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
        x = self.se(x) if self.se else x
        x = self.mamba(x) if self.mamba else x
        x = self.vit(x) if self.vit else x
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


class MyNetB(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        depth=3,
        encoder_channels=[48, 96, 192, 384],
        conv=DefineConv,
        deep_supervision=False,
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
        # self.encoders.append(DefineConv(in_channels, encoder_channels[0]))
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
                    se=True,
                    mamba=True if i > self.depth - 1 else False,
                    vit=False,
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
                    se=True,
                    mamba=True if i < 0 else False,
                    vit=False,
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
