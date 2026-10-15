"""Raydrop refinement: the LiDAR4D U-Net on a frozen model's rendered maps.

LiDAR4D, GS-LiDAR, SN-LiDAR and LiDAR-RT all refine raydrop with the same
network: GS-LiDAR's ``scene/unet.py`` is byte-identical to LiDAR4D's
``model/unet.py``, SN-LiDAR drops one docstring and LiDAR-RT only reformats it.
``UNet`` below is that file's network, unchanged. Each of those methods trains
it after its scene model has finished, with the scene model frozen, and at test
time replaces only the raydrop mask; depth and intensity values stay the
frozen model's.

``RaydropRefiner`` adapts the LiDAR4D/GS-LiDAR input recipe to this repo's
renders: three channels [raydrop, intensity, depth / depth_scale_m], with
raydrop in this repo's convention (1 = no return, GS-LiDAR's as well), and the
median depth because that is the depth the evaluation reports. During training
the input gets LiDAR4D's random zero boxes.
"""
# ==============================================================================
# UNet and its blocks: Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D (model/unet.py at 4d6abbd)
# Apache License 2.0
# ==============================================================================
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """
    (BN => ReLU => Dropout => Conv) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttnBlock(nn.Module):
    """
    Multi-head attention with dropout
    """
    def __init__(self, in_ch, num_head=8, dropout=0.1):
        super().__init__()
        self.proj_qkv = nn.Conv2d(in_ch, in_ch * 3, 1, stride=1, padding=0, bias=False)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(in_ch)
        self.dropout = dropout
        self.num_head = num_head

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.proj_qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # multi-head attention
        q = q.view(B, self.num_head, -1, H * W).permute(0, 1, 3, 2)  # [B, num_head, H * W, C]
        k = k.view(B, self.num_head, -1, H * W)  # [B, num_head, C, H * W]
        v = v.view(B, self.num_head, -1, H * W).permute(0, 1, 3, 2)  # [B, num_head, H * W, C]

        w = torch.matmul(q, k) * (int(C//self.num_head) ** (-0.5))

        # dropout
        if self.training:
            m_r = torch.ones_like(w) * self.dropout
            w = w + torch.bernoulli(m_r) * -1e12

        w = F.softmax(w, dim=-1)
        h = torch.matmul(w, v)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        return x + h


class InConv(nn.Module):
    """
    First layer conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    """
    Last layer conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Efficient U-Net of LiDAR4D for Ray-drop Refinement
    """
    def __init__(self, in_channels, channels=32, out_channels=1):
        super(UNet, self).__init__()
        self.inc = InConv(in_channels, channels)
        self.down1 = Down(channels, channels*2)     # Down(32, 64)
        self.down2 = Down(channels*2, channels*4)   # Down(64, 128)
        self.down3 = Down(channels*4, channels*8)   # Down(128, 256)
        self.down4 = Down(channels*8, channels*8)   # Down(256, 256)
        self.attn = AttnBlock(channels*8)
        self.up1 = Up(channels*16, channels*4)      # Up(512, 128)
        self.up2 = Up(channels*8, channels*2)       # Up(256, 64)
        self.up3 = Up(channels*4, channels)         # Up(128, 32)
        self.up4 = Up(channels*2, channels)         # Up(64, 32)
        self.outc = OutConv(channels, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x0 = self.inc(input)                        # 3 -> 32
        x1 = self.down1(x0)                         # 32 -> 64
        x2 = self.down2(x1)                         # 64 -> 128
        x3 = self.down3(x2)                         # 128 -> 256
        x4 = self.down4(x3)                         # 256 -> 256
        x4 = self.attn(x4)
        x_out = self.up1(x4, x3)                    # 256+256 -> 128
        x_out = self.up2(x_out, x2)                 # 128+128 -> 64
        x_out = self.up3(x_out, x1)                 # 64+64 -> 32
        x_out = self.up4(x_out, x0)                 # 32+32 -> 32
        logits = self.outc(x_out)                   # 32 -> 1
        outputs = self.sigmoid(logits)
        return outputs


# Render keys of the three input channels, in LiDAR4D's channel order.
REFINE_INPUT_KEYS = ("raydrop", "intensity_sh", "depth_median")

# No-return probability above this is a dropped ray, as in Loss and the
# LiDAR4D/GS-LiDAR evaluation.
RAYDROP_THRESHOLD = 0.5


def raydrop_scores(pred, gt, view_mask):
    """BCE (with gradient) and detached acc/F1/RMSE over real views' pixels.

    ``pred``/``gt`` are ``[B, V, 1, H, W]`` no-return maps (1 = dropped);
    padded views are excluded. F1 treats a dropped ray as the positive class,
    like ``raydrop_errors`` in the evaluation.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"raydrop maps differ: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    pixels = view_mask.to(device=pred.device, dtype=torch.bool)
    p = pred[pixels].reshape(-1)
    g = gt[pixels].reshape(-1).to(p.dtype)
    scores = {"loss_raydrop": F.binary_cross_entropy(p.clamp(1e-7, 1.0 - 1e-7), g)}
    with torch.no_grad():
        dropped = p > RAYDROP_THRESHOLD
        truth = g > 0.5
        tp = (dropped & truth).sum().to(p.dtype)
        fp = (dropped & ~truth).sum().to(p.dtype)
        fn = (~dropped & truth).sum().to(p.dtype)
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        scores["raydrop_acc"] = (dropped == truth).to(p.dtype).mean()
        scores["raydrop_f1"] = torch.where(
            precision + recall > 0,
            2.0 * precision * recall / (precision + recall).clamp_min(1e-12),
            torch.zeros_like(precision),
        )
        scores["raydrop_rmse"] = (p - g).square().mean().sqrt()
    return scores


class RaydropRefiner(nn.Module):
    """Refine a render batch's raydrop maps with the LiDAR4D U-Net.

    ``forward`` takes the renderer's output dict (``[B, V, 1, H, W]`` maps
    plus ``view_mask``) and returns refined no-return probabilities of the same
    shape. Only real views enter the U-Net, so the zero-padded views of short
    windows never reach BatchNorm; their rows keep the input raydrop.
    """

    def __init__(self, cfg):
        super().__init__()
        self.depth_scale_m = float(cfg.depth_scale_m)
        box_mask = cfg.box_mask
        self.box_mask = bool(box_mask.enable)
        self.box_max_count = int(box_mask.max_boxes)
        self.box_max_frac = float(box_mask.max_size_frac)
        self.unet = UNet(
            in_channels=len(REFINE_INPUT_KEYS),
            channels=int(cfg.channels),
            out_channels=1,
        )

    def unet_input(self, renders):
        """``[N, 3, H, W]`` channels of every real view, and the view mask."""
        raydrop, intensity, depth = (renders[key] for key in REFINE_INPUT_KEYS)
        if raydrop.dim() != 5 or raydrop.shape[2] != 1:
            raise ValueError(
                "raydrop refinement expects [B, V, 1, H, W] render maps, got "
                f"{tuple(raydrop.shape)}"
            )
        view_mask = renders.get("view_mask")
        if view_mask is None:
            view_mask = torch.ones(
                raydrop.shape[:2], dtype=torch.bool, device=raydrop.device
            )
        view_mask = view_mask.to(device=raydrop.device, dtype=torch.bool)
        stacked = torch.cat(
            [raydrop, intensity, depth / self.depth_scale_m], dim=2
        )
        return stacked[view_mask], view_mask

    def random_box_mask(self, x):
        """LiDAR4D's training augmentation: one draw of 0-31 zeroed boxes,
        each up to 10% of H and W, shared by every image and channel."""
        mask = torch.ones_like(x)
        height, width = x.shape[-2:]
        max_h = max(2, int(self.box_max_frac * height))
        max_w = max(2, int(self.box_max_frac * width))
        count = int(torch.randint(self.box_max_count, ()).item())
        for _ in range(count):
            box_h = int(torch.randint(1, max_h, ()).item())
            box_w = int(torch.randint(1, max_w, ()).item())
            top = int(torch.randint(height - box_h, ()).item())
            left = int(torch.randint(width - box_w, ()).item())
            mask[..., top:top + box_h, left:left + box_w] = 0.0
        return x * mask

    def forward(self, renders):
        x, view_mask = self.unet_input(renders)
        if self.training and self.box_mask:
            x = self.random_box_mask(x)
        refined = self.unet(x)
        out = renders["raydrop"].clone()
        out[view_mask] = refined.to(out.dtype)
        return out


__all__ = ["RAYDROP_THRESHOLD", "REFINE_INPUT_KEYS", "RaydropRefiner", "UNet", "raydrop_scores"]
