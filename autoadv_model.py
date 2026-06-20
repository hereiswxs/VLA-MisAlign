import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 频域信息提取模块
# ----------------------------
class FrequencyDomainExtractor(nn.Module):
    def __init__(self, high_freq_cutoff=0.1):
        super().__init__()
        self.high_freq_cutoff = high_freq_cutoff

    def forward(self, x):
        # 进行FFT变换，将图像从空间域转换到频域
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x, dim=[2, 3])
        
        # 提取频域信息，移除低频部分（通过设置 cutoff 参数）
        cutoff_idx_h = int(H * self.high_freq_cutoff)
        cutoff_idx_w = int(W * self.high_freq_cutoff)

        # 高频区域
        x_fft[:, :, :cutoff_idx_h, :cutoff_idx_w] = 0
        x_fft[:, :, -cutoff_idx_h:, :cutoff_idx_w] = 0
        x_fft[:, :, :cutoff_idx_h, -cutoff_idx_w:] = 0
        x_fft[:, :, -cutoff_idx_h:, -cutoff_idx_w:] = 0
        
        # 转换回空间域
        x_ifft = torch.fft.ifft2(x_fft, dim=[2, 3])
        return x_ifft.real  # 只保留实部

# ----------------------------
# 频域注意力模块
# ----------------------------
class FrequencyAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, 1, 3, padding=1)  # 输出单通道的注意力图
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算频域信息的注意力图
        attn_map = self.sigmoid(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return x * attn_map  # 应用注意力图



# ----------------------------
# 空间 + 通道注意力（CBAM）
# ----------------------------
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ch_attn = self.channel_attn(x)
        x = x * ch_attn

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_attn = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        x = x * s_attn

        return x

# ----------------------------
# Cross-Attention 模块
# ----------------------------

class CrossAttention(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim):
        super().__init__()
        self.task_proj = nn.Linear(text_dim, visual_dim)  # 4096 -> 256
        self.q = nn.Linear(visual_dim, hidden_dim)
        self.k = nn.Linear(visual_dim, hidden_dim)
        self.v = nn.Linear(visual_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, visual_dim)

    def forward(self, visual_feat, text_embed):  # visual_feat: [B,C,H,W], text_embed: [B,D]
        B, C, H, W = visual_feat.shape
        v_flat = visual_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        q = self.q(v_flat)             # [B, HW, H]

        task_feat = self.task_proj(text_embed)  # [B, visual_dim]
        k = self.k(task_feat).unsqueeze(1)      # [B, 1, H]
        v = self.v(task_feat).unsqueeze(1)      # [B, 1, H]

        attn = torch.softmax((q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5), dim=-1)  # [B, HW, 1]
        out = attn @ v  # [B, HW, H]
        out = self.proj(out).permute(0, 2, 1).view(B, C, H, W)
        return visual_feat + out


class MultiModalAttention(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim):
        super().__init__()
        self.cross_attention = CrossAttention(visual_dim, text_dim, hidden_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=8)
        self.proj = nn.Linear(visual_dim, visual_dim)

    def forward(self, visual_feat, text_embed):
        # 使用跨模态注意力融合视觉和文本信息
        visual_feat = self.cross_attention(visual_feat, text_embed)
        
        # 对视觉特征进行自注意力处理
        B, C, H, W = visual_feat.shape
        visual_flat = visual_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        
        # 使用 MultiheadAttention 计算自注意力
        attn_output, _ = self.self_attention(visual_flat, visual_flat, visual_flat)  # 自注意力
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)  # 还原为 [B, C, H, W]
        
        # 投影回视觉维度
        # 展平为 [B * HW, C]
        attn_output_flat = attn_output.reshape(B * H * W, C)  # 使用 reshape 替代 view
        attn_output_proj = self.proj(attn_output_flat)  # 投影到目标维度
        attn_output_proj = attn_output_proj.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # 恢复为 [B, visual_dim, H, W]

        return attn_output_proj


# ----------------------------
# 基础残差块
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

# ----------------------------
# 编码器
# ----------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),  # downsample
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels),
            CBAMBlock(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

# ----------------------------
# 解码器（支持跳跃连接）
# ----------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, output_padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.res = ResidualBlock(out_channels)
        self.attn = CBAMBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.relu(self.bn(self.up(x)))
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = x + skip
        x = self.res(x)
        x = self.attn(x)
        return x

# ----------------------------
# 主体扰动生成器
# ----------------------------
class PerturbationGenerator(nn.Module):
    def __init__(self, input_channels=3, task_dim=4096):
        super().__init__()

        # 编码器部分
        self.enc1 = EncoderBlock(input_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # 融合语义的 Cross-Attention
        self.multi_modal_attn = MultiModalAttention(visual_dim=512, text_dim=task_dim, hidden_dim=512)

        # 中间残差块
        self.middle = ResidualBlock(512)

        # 频域信息提取与频域注意力模块
        self.freq_extractor = FrequencyDomainExtractor(high_freq_cutoff=0.1)
        self.freq_attention = FrequencyAttentionBlock(512)



        # 解码器部分
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)

        # 最终 refinement
        self.refine = nn.Sequential(
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, input_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )


    def forward(self, x, task_embed):  # x: [B,3,H,W]; task_embed: [B,D]
        # 频域信息提取与频域注意力
        freq_info = self.freq_extractor(x)  # 提取频域信息
        #print(m.shape)
        #print(freq_info.shape)
        x = x + freq_info  # 将频域信息融合到生成特征中

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 融合任务语义
        e4 = self.multi_modal_attn(e4, task_embed)

        # 中间层
        m = self.middle(e4)


        m_with_freq = self.freq_attention(m)  # 使用频域注意力加强高频区域

        # 解码 + 跳连
        d4 = self.dec4(m_with_freq, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        out = self.refine(d2)

        # 插值确保大小一致
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 返回 residual 风格扰动
        return out,freq_info,freq_info
