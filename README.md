# hdsnet code
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================================================================
# 1. 基础组件 (ResBlock, ASPP, DoubleDown)
# ==================================================================

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 2, 4, 6]
        self.aspp_blocks = nn.ModuleList()
        for d in dilations:
            self.aspp_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
            ))
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.5)
        )

    def forward(self, x):
        res = [block(x) for block in self.aspp_blocks]
        global_feat = F.interpolate(self.global_avg_pool(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(global_feat)
        return self.project(torch.cat(res, dim=1))


class DoubleDownSampling(nn.Module):
    def __init__(self, in_c=3, out_c=32):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, out_c, 3, 2, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU()
        )

    def forward(self, x): return self.down(x)


# ==================================================================
# 2. 核心模块：GDABlock (CNN Attention) - 已更新 (移除 Beta 相加项)
# ==================================================================

class GDABlock(nn.Module):
    """ [Ours] Grouped Dual Attention (Optimized) """

    def __init__(self, in_channels, group_n=8, window_size=7, use_dwconv=True, **kwargs):
        super(GDABlock, self).__init__()

        assert in_channels % group_n == 0, f"Input channels ({in_channels}) must be divisible by group_n ({group_n})"

        self.in_channels = in_channels
        self.group_n = group_n
        self.c_per_group = in_channels // group_n

        self.conv1x1_group = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=group_n, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.use_dwconv = use_dwconv
        if self.use_dwconv:
            self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=window_size, padding=(window_size - 1) // 2,
                                groups=in_channels, bias=False)
            self.bn2 = nn.BatchNorm2d(in_channels)
        else:
            self.dw = None
            self.bn2 = None

        self.act = nn.SiLU(inplace=True)

        k = 3
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sp_conv = nn.Conv2d(self.group_n * 2, self.group_n, kernel_size=3, padding=1, groups=self.group_n,
                                 bias=False)

        # 仅保留 alpha
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 预处理
        x_grouped = self.conv1x1_group(x)
        x_grouped = self.bn1(x_grouped)

        # 2. 残差连接
        if self.use_dwconv:
            residual = self.dw(x_grouped)
            residual = self.bn2(residual)
            x_grouped = x_grouped + residual

        # 3. 激活
        x_grouped = self.act(x_grouped)

        # --- Attention ---
        x_split = x_grouped.view(B, self.group_n, self.c_per_group, H, W)
        x_group_avg = x_split.mean(dim=2)

        # Channel Attn
        ch_desc = F.adaptive_avg_pool2d(x_group_avg, (1, 1)).view(B, self.group_n, 1).transpose(1, 2)
        ch_att = torch.sigmoid(self.eca_conv(ch_desc).transpose(1, 2).view(B, self.group_n, 1, 1)).unsqueeze(2)

        # Spatial Attn
        group_max, _ = x_split.max(dim=2)
        group_mean = x_split.mean(dim=2)
        sp_att = torch.sigmoid(self.sp_conv(torch.cat([group_max, group_mean], dim=1))).unsqueeze(2)

        # 移除 beta 加法项，仅保留 alpha 的乘法项
        fused_weight = torch.sigmoid(self.alpha) * (ch_att * sp_att)
        return (x_split * fused_weight).reshape(B, C, H, W)


# ==================================================================
# 3. 核心模块：LGF-Block (Transformer)
# ==================================================================

class LGF_Block_Spatial(nn.Module):
    """ [Ours] Spatial LGF Block - 改进版交叉通道注意力融合 + Alpha动态加权 """

    def __init__(self, in_channels, embed_dim=None, num_heads=4, mlp_ratio=2.0, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        embed_dim = in_channels if embed_dim is None else embed_dim

        # 1. Conv Branch
        self.edge_spatial = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1), groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0), groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # 2. Trans Branch Prep
        raw_dim = in_channels * patch_size * patch_size
        self.patch_reduce = nn.Sequential(nn.Linear(raw_dim, embed_dim), nn.LayerNorm(embed_dim))

        # 3. Attention & MLP
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

        # 4. Restore
        self.pixel_shuffle = nn.PixelShuffle(patch_size)
        self.beta = nn.Parameter(torch.ones(1))

        # ==================== 融合模块准备 ====================

        out_c_trans = embed_dim // (patch_size * patch_size)
        if out_c_trans != in_channels:
            self.align_conv = nn.Conv2d(out_c_trans, in_channels, kernel_size=1, bias=False)
            self.align_norm = nn.BatchNorm2d(in_channels)
        else:
            self.align_conv = nn.Identity()
            self.align_norm = nn.Identity()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attn_conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, bias=False)
        self.sigmoid = nn.Sigmoid()


        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size


        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        x_padded = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h > 0 or pad_w > 0) else x

        B_pad, C_pad, H_pad, W_pad = x_padded.shape


        x_conv = self.edge_spatial(x_padded)

        # --- Path 2: Trans Branch ---
        x_reshaped = x_padded.view(B_pad, C_pad, H_pad // p, p, W_pad // p, p)
        x_permuted = x_reshaped.permute(0, 2, 4, 1, 3, 5)
        x_patches = x_permuted.contiguous().reshape(B_pad, -1, C_pad * p * p)

        x_emb = self.patch_reduce(x_patches)
        x_attn, _ = self.attn(self.norm1(x_emb), self.norm1(x_emb), self.norm1(x_emb))
        x_trans = x_emb + self.beta * x_attn
        x_trans = x_trans + self.mlp(self.norm2(x_trans))

        # Restore
        h_tokens, w_tokens = H_pad // p, W_pad // p
        x_trans_spatial = x_trans.transpose(1, 2).reshape(B_pad, -1, h_tokens, w_tokens)
        x_restored = self.pixel_shuffle(x_trans_spatial)
        x_restored = self.align_norm(self.align_conv(x_restored))


        gap_conv = self.gap(x_conv).view(B_pad, C_pad)
        gap_trans = self.gap(x_restored).view(B_pad, C_pad)


        interleaved = torch.stack([gap_conv, gap_trans], dim=2).reshape(B_pad, 1, 2 * C_pad)
        attn_weight = self.attn_conv1d(interleaved)
        attn_weight = self.sigmoid(attn_weight).view(B_pad, C_pad, 1, 1)

        alpha_val = torch.sigmoid(self.alpha)

        x_conv_new = x_conv * (alpha_val * attn_weight)
        x_trans_new = x_restored * ((1.0 - alpha_val) * attn_weight)

        out = x_conv_new + x_trans_new


        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out

# ==================================================================
# 4. 核心模块：LKA-Merging (Downsample)
# ==================================================================

class LKA_Merging_Spatial(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()

        # 自动计算 padding
        pad_size = kernel_size // 2

        # Weight Generator
        self.weight_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=2, padding=pad_size,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Projection
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_pooled = self.pool(x)
        x_weight = self.weight_gen(x)
        x_out = x_pooled * (1 + x_weight)
        return self.norm(self.proj(x_out))


# ==================================================================
# 5. 融合模块 (BiFFM)
# ==================================================================

class BiFFM(nn.Module):
    def __init__(self, t_dim, c_dim, out_dim):
        super().__init__()
        self.t_proj = nn.Conv2d(t_dim, out_dim, kernel_size=1, bias=False)
        self.c_proj = nn.Conv2d(c_dim, out_dim, kernel_size=1, bias=False)
        self.bn_t = nn.BatchNorm2d(out_dim)
        self.bn_c = nn.BatchNorm2d(out_dim)
        self.spatial_attn = nn.Sequential(nn.Conv2d(out_dim * 2, 1, kernel_size=7, padding=3), nn.Sigmoid())
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)
        )

    def forward(self, t_feat, c_feat):
        t_x = F.relu(self.bn_t(self.t_proj(t_feat)))
        c_x = F.relu(self.bn_c(self.c_proj(c_feat)))
        if t_x.shape[2:] != c_x.shape[2:]:
            t_x = F.interpolate(t_x, size=c_x.shape[2:], mode='bilinear', align_corners=False)
        s_map = self.spatial_attn(torch.cat([t_x, c_x], dim=1))
        return self.fusion_conv(torch.cat([t_x * s_map, c_x], dim=1))


# ==================================================================
# 6. 主网络：HDSNet
# ==================================================================

class HDSNet(nn.Module):
    def __init__(self, in_channels=3,num_classes=4):
        super().__init__()

        print("Init HDSNet (Full Version) | Using GDABlock(No Beta), LGF-Block(Cross-Attn), LKA-Merging")

        # --- CNN Branch ---
        self.c_stage1 = nn.Sequential(ResBlock(in_channels, 32, stride=2), GDABlock(32, group_n=4))
        self.c_stage2 = nn.Sequential(ResBlock(32, 64, stride=2), GDABlock(64, group_n=8))
        self.c_stage3 = nn.Sequential(ResBlock(64, 128, stride=2), GDABlock(128, group_n=16))

        # --- Transformer Branch ---
        self.double_down = DoubleDownSampling(in_channels, 32)

        # Stage 1 (Input: 32)
        self.trans_stage1 = nn.Sequential(
            LGF_Block_Spatial(in_channels=32, embed_dim=64, patch_size=4),
            LGF_Block_Spatial(in_channels=32, embed_dim=64, patch_size=4)
        )
        self.merge1 = LKA_Merging_Spatial(in_channels=32, out_channels=64, kernel_size=7)

        # Stage 2 (Input: 64)
        self.trans_stage2 = nn.Sequential(
            LGF_Block_Spatial(in_channels=64, embed_dim=128, patch_size=4),
            LGF_Block_Spatial(in_channels=64, embed_dim=128, patch_size=4)
        )
        self.merge2 = LKA_Merging_Spatial(in_channels=64, out_channels=128, kernel_size=7)

        # Stage 3 (Input: 128)
        self.trans_stage3 = nn.Sequential(
            LGF_Block_Spatial(in_channels=128, embed_dim=256, patch_size=4),
            LGF_Block_Spatial(in_channels=128, embed_dim=256, patch_size=4)
        )

        # --- Fusion & Decoder ---
        self.ffm1 = BiFFM(t_dim=32, c_dim=32, out_dim=64)
        self.ffm2 = BiFFM(t_dim=64, c_dim=64, out_dim=128)
        self.ffm3 = BiFFM(t_dim=128, c_dim=128, out_dim=128)

        self.aspp = ASPP(128, 128)
        self.up3 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Sequential(nn.Conv2d(128 + 128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up1 = nn.Sequential(nn.Conv2d(64 + 64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(32, 16, 3, padding=1),
                                   nn.BatchNorm2d(16), nn.ReLU())
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        # CNN Forward
        c1 = self.c_stage1(x)
        c2 = self.c_stage2(c1)
        c3 = self.c_stage3(c2)

        # Transformer Forward (Spatial Flow)
        t_in = self.double_down(x)  # [B, 32, H/4, W/4]

        t1 = self.trans_stage1(t_in)
        t1_down = self.merge1(t1)  # Downsample -> [B, 64, H/8, W/8]

        t2 = self.trans_stage2(t1_down)
        t2_down = self.merge2(t2)  # Downsample -> [B, 128, H/16, W/16]

        t3 = self.trans_stage3(t2_down)

        # Fusion
        f1 = self.ffm1(t1, c1)
        f2 = self.ffm2(t2, c2)
        f3 = self.ffm3(t3, c3)

        # Decoder
        x_aspp = self.aspp(f3)
        d3 = self.up3(x_aspp)
        d3_up = F.interpolate(d3, size=f2.shape[2:], mode='bilinear')
        d2 = self.up2(torch.cat([d3_up, f2], dim=1))

        d2_up = F.interpolate(d2, size=f1.shape[2:], mode='bilinear')
        d1 = self.up1(torch.cat([d2_up, f1], dim=1))

        return self.head(self.final(d1))


# ==================================================================
# 7. 运行测试
# ==================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Test 1: Full Model (Ours) ---")
    model = HDSNet().to(device)
    x = torch.randn(2, 3, 800, 160).to(device)
    out = model(x)
    print(f"Input: 800x160, Output: {out.shape}")

    print("\n--- Test 2: Extreme Size (1024x64) ---")
    x_ex = torch.randn(2, 3, 1024, 160).to(device)
    out_ex = model(x_ex)
    print(f"Extreme Input Output: {out_ex.shape}")
