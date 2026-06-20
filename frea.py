import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 定义 FrequencyDomainExtractor ==========
class FrequencyDomainExtractor(nn.Module):
    """
    平滑带通 + 分位数去背景 + 可选DoG；输出单通道 log-幅度边缘图（未归一）。
    """
    def __init__(self, high_pass: float = 0.2, low_pass: float = None,
                 smooth: float = 10.0,  # 过渡平滑度
                 qbg: float = 0.01,     # 背景分位数阈
                 use_dog: bool = True,
                 eps: float = 1e-8):
        super().__init__()
        assert 0.0 <= high_pass <= 1.0
        if low_pass is not None:
            assert 0.0 < high_pass < low_pass <= 1.0
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.smooth = smooth
        self.qbg = qbg
        self.use_dog = use_dog
        self.eps = eps
        self._rr_cache = {}

    def _get_rr(self, H: int, W: int, device, dtype):
        key = (H, W, device, dtype)
        rr = self._rr_cache.get(key, None)
        if rr is None:
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
                torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
                indexing="ij",
            )
            rr = torch.sqrt(xx**2 + yy**2) / (2 ** 0.5)  # 0~1
            self._rr_cache[key] = rr
        return rr

    @staticmethod
    def _smooth_bandpass(rr, hp, lp, s):
        if lp is None:  # 仅高通
            return torch.sigmoid((rr - hp) * s)
        up = torch.sigmoid((rr - hp) * s)
        down = torch.sigmoid((lp - rr) * s)
        return up * down

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x, dim=[2, 3])
        x_fft = torch.fft.fftshift(x_fft, dim=[2, 3])

        rr = self._get_rr(H, W, x.device, x.dtype)
        mask = self._smooth_bandpass(rr, self.high_pass, self.low_pass, self.smooth)
        x_fft_bp = x_fft * mask.view(1, 1, H, W)

        x_fft_bp = torch.fft.ifftshift(x_fft_bp, dim=[2, 3])
        x_ifft = torch.fft.ifft2(x_fft_bp, dim=[2, 3])
        mag = torch.abs(x_ifft).mean(dim=1, keepdim=True)
        freq_edge = torch.log(mag + self.eps)

        # 分位数去背景
        q = torch.quantile(freq_edge.view(B, -1), self.qbg, dim=1).view(B, 1, 1, 1)
        freq_edge = torch.relu(freq_edge - q)

        # DoG 增强
        if self.use_dog:
            blur = F.avg_pool2d(freq_edge, kernel_size=3, stride=1, padding=1)
            freq_edge = torch.relu(freq_edge - blur)

        return freq_edge  # [B,1,H,W]

# ========== 读取图像 ==========
img_path = "/home/student/DongXiaorong/Madv_VLA/datasets/libero_object_no_noops/pick_up_the_ketchup_and_place_it_in_the_basket_demo/demo_7/frame_005.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 转成Tensor [B,C,H,W]
x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0

# ========== 提取频域高频图 ==========
extractor = FrequencyDomainExtractor(high_pass=0.2, low_pass=0.8, qbg=0.01, use_dog=True)
freq_edge = extractor(x)  # [1,1,H,W]
freq_edge_np = freq_edge.squeeze().cpu().numpy()

# 归一化到0-255
img_norm = cv2.normalize(freq_edge_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, mask_bin = cv2.threshold(img_norm, 20, 255, cv2.THRESH_BINARY)

# ========== 连通域分析 ==========
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
min_area = 200
mask_clean = np.zeros_like(mask_bin)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > min_area:
        mask_clean[labels == i] = 255


# ========== 保存 ==========
cv2.imwrite("foreground_mask_raw.png", mask_bin)
cv2.imwrite("foreground_mask_clean.png", mask_clean)
cv2.imwrite("foreground_enhanced.png", img_norm)

print("结果已保存：foreground_mask_raw.png, foreground_mask_clean.png, foreground_enhanced.png")
