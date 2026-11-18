from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .vmamba import VSSBlockBCHW, ConditionalVSSBlockBCHW, FrequencyVSSBlockBCHW


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, act="SiLU"):
        super().__init__()
        assert act in ["None", "ReLU", "GELU", "SiLU"], f"Unsupported activation: {act}"
        if p is None:
            p = k // 2
        if act == "ReLU":
            activation = nn.ReLU(inplace=True)
        elif act == "GELU":
            activation = nn.GELU()
        elif act == "SiLU":
            activation = nn.SiLU(inplace=True)
        else:
            activation = nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            activation,
        )

    def forward(self, x):
        return self.net(x)


def fft2_mag_phase(x: torch.Tensor, eps: float = 1e-8):
    """BCHW real tensor -> centered FFT magnitude and phase."""
    spec = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
    spec = torch.fft.fftshift(spec, dim=(-2, -1))
    mag = spec.abs().clamp_min(eps)
    phase = torch.angle(spec)
    return mag.to(dtype=x.dtype), phase.to(dtype=x.dtype)


def ifft2_from_mag_phase(mag: torch.Tensor, phase: torch.Tensor):
    spec = torch.polar(mag.float(), phase.float())
    spec = torch.fft.ifftshift(spec, dim=(-2, -1))
    x = torch.fft.ifft2(spec, dim=(-2, -1), norm="ortho").real
    return x.to(dtype=mag.dtype)


def gaussian_masks(h, w, device, dtype, sigma_ratio=0.18):
    # sigma 控制低频区域的范围
    yy = torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2
    xx = torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    sigma = max(float(min(h, w)) * sigma_ratio, 1.0)
    low = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma)).view(1, 1, h, w)
    return low, 1.0 - low


class FrequencyStem(nn.Module):
    """Resize + 1x1 projection for the first frequency stage."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0, act="ReLU")

    def forward(self, x, size, downsample="Avg"):
        assert downsample in ["Avg", "Max", "Bilinear"], f"Unsupported downsample: {downsample}"
        if x.shape[-2:] != size:
            if downsample == "Avg":
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
            elif downsample == "Max":
                x = F.max_pool2d(x, kernel_size=2, stride=2)
            else:
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if  x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.proj(x)
    

class FrequencyDown(nn.Module):
    """Resize + 1x1 channel transform between frequency stages."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0)

    def forward(self, x, size, downsample="Avg"):
        assert downsample in ["Avg", "Max", "Bilinear"], f"Unsupported downsample: {downsample}"
        if x.shape[-2:] != size:
            if downsample == "Avg":
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
            elif downsample == "Max":
                x = F.max_pool2d(x, kernel_size=2, stride=2)
            else:
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if  x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.proj(x)


class FrequencyEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int,
        dims: Sequence[int],
        depths: Sequence[int] = (1, 1, 1, 1),
        ssm_d_state: int = 16,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        assert len(dims) == 4 and len(depths) == 4
        self.stem = FrequencyStem(in_ch, dims[0])
        self.downs = nn.ModuleList([FrequencyDown(dims[i - 1], dims[i]) for i in range(1, 4)])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        self.spatial_fft_fuse = nn.ModuleList()

        for i, dim in enumerate(dims):
            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    FrequencyVSSBlockBCHW(
                        hidden_dim=dim,
                        drop_path=dpr[cur + j],
                        d_state=ssm_d_state,
                    )
                )

            cur += depths[i]
            self.stages.append(nn.Sequential(*blocks))
            self.spatial_fft_fuse.append(
                nn.Sequential(
                    ConvBNAct(dim * 2, dim, k=1, p=0, act="ReLU"),
                    ConvBNAct(dim, dim, k=1, p=0, act="ReLU"),
                )
            )

    def forward(self, image_like: torch.Tensor, spatial_feats: Sequence[torch.Tensor]):
        img_mag, _ = fft2_mag_phase(image_like)
        img_mag = torch.log1p(img_mag)

        freq_feats: List[torch.Tensor] = []
        spatial_phases: List[torch.Tensor] = []

        x = self.stem(img_mag, spatial_feats[0].shape[-2:]) # [B, dims[0], H/4, W/4]

        for i in range(4):
            if i > 0:
                # 上一层的频域特征下采样到当前层尺寸
                x = self.downs[i - 1](x, spatial_feats[i].shape[-2:])

            x = self.stages[i](x)

            sp_mag, sp_phase = fft2_mag_phase(spatial_feats[i])
            spatial_phases.append(sp_phase)
            sp_mag = torch.log1p(sp_mag)

            x = self.spatial_fft_fuse[i](torch.cat([x, sp_mag], dim=1))
            freq_feats.append(x)

        return freq_feats, spatial_phases


class FrequencyBandToSpace(nn.Module):
    """frequency feature -> Gaussian low/high bands -> IFFT with spatial phase."""
    def __init__(self, channels: int, sigma_ratio: float = 0.18):
        super().__init__()
        self.sigma_ratio = sigma_ratio
        self.low_refine = ConvBNAct(channels, channels, k=3, p=1, act="SiLU")
        self.high_refine = ConvBNAct(channels, channels, k=3, p=1, act="SiLU")

    def forward(self, freq_feature: torch.Tensor, spatial_phase: torch.Tensor):
        mag = F.softplus(freq_feature)
        h, w = mag.shape[-2:]
        low_mask, high_mask = gaussian_masks(h, w, mag.device, mag.dtype, self.sigma_ratio)
        low = ifft2_from_mag_phase(mag * low_mask, spatial_phase)
        high = ifft2_from_mag_phase(mag * high_mask, spatial_phase)
        return self.low_refine(low), self.high_refine(high)


class OneWayCrossDomainFigC(nn.Module):
    """
    One-way cross-modal cross-domain fusion.

    Example:
        target_spatial = IR_S
        cond_low       = RGB_LP
        cond_high      = RGB_HP

    Then:
        RGB_LP guides IR_S through one Conditional VSS block.
        RGB_HP guides IR_S through another Conditional VSS block.
    """
    def __init__(self, dim: int, d_state: int = 16, drop_path: float = 0.0):
        super().__init__()

        self.low_block = ConditionalVSSBlockBCHW(
            hidden_dim=dim,
            drop_path=drop_path,
            d_state=d_state,
        )

        self.high_block = ConditionalVSSBlockBCHW(
            hidden_dim=dim,
            drop_path=drop_path,
            d_state=d_state,
        )

        self.low_linear = ConvBNAct(dim, dim, k=3, p=1, act="SiLU")
        self.high_linear = ConvBNAct(dim, dim, k=3, p=1, act="SiLU")
        self.out_linear = ConvBNAct(dim, dim, k=3, p=1, act="SiLU")

    def forward(self, target_spatial, cond_low, cond_high):
        low_update = self.low_linear(
            self.low_block(target_spatial, cond_low, cond_low)
        )

        high_update = self.high_linear(
            self.high_block(target_spatial, cond_high, cond_high)
        )

        update = self.out_linear(low_update + high_update)

        return target_spatial + update


class CrossModalCrossDomainFigC(nn.Module):
    """
    RGB_S is guided by IR_LP / IR_HP.
    IR_S is guided by RGB_LP / RGB_HP.
    """
    def __init__(self, dim: int, d_state: int = 16, drop_path: float = 0.0):
        super().__init__()

        self.rgb_by_ir = OneWayCrossDomainFigC(
            dim=dim,
            d_state=d_state,
            drop_path=drop_path,
        )

        self.ir_by_rgb = OneWayCrossDomainFigC(
            dim=dim,
            d_state=d_state,
            drop_path=drop_path,
        )

    def forward(self, rgb_s, ir_s, rgb_low, rgb_high, ir_low, ir_high):
        rgb_sf = self.rgb_by_ir(rgb_s, ir_low, ir_high)
        ir_sf = self.ir_by_rgb(ir_s, rgb_low, rgb_high)

        return rgb_sf, ir_sf


class MultiScaleConv(nn.Module):
    """Parallel 1x1, 3x3, 5x5, 7x7 multi-scale feature extraction."""
    def __init__(self, dim: int):
        super().__init__()
        self.branch1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.branch5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, bias=False)
        self.branch7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.fuse = ConvBNAct(dim * 4, dim, k=1, p=0, act="SiLU")

    def forward(self, x):
        x1 = self.branch1(x)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)

        x = torch.cat([x1, x3, x5, x7], dim=1)
        x = self.fuse(x)
        return x
    
class DifferenceGuidedCrossLevel(nn.Module):
    """Difference-modulated cross-level decoder. """
    def __init__(self, dim: int, prev_dim: Optional[int], num_classes: int = 1):
        super().__init__()

        self.rgb_ms = MultiScaleConv(dim)
        self.ir_ms = MultiScaleConv(dim)
        self.diff_gate = nn.Sequential(
            ConvBNAct(dim, dim, k=3, p=1, act="SiLU"),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 修改
        # 跨层融合：RGB 和 IR 分开融合
        # 当前层: dim
        # 上一深层: prev_dim
        # concat 后: dim + prev_dim
        self.rgb_prev_fuse = None
        self.ir_prev_fuse = None

        if prev_dim is not None:
            self.rgb_prev_fuse = nn.Sequential(
                ConvBNAct(dim + prev_dim, dim, k=1, p=0, act="SiLU"),
                ConvBNAct(dim, dim, k=3, act="SiLU"),
            )

            self.ir_prev_fuse = nn.Sequential(
                ConvBNAct(dim + prev_dim, dim, k=1, p=0, act="SiLU"),
                ConvBNAct(dim, dim, k=3, act="SiLU"),
            )

        self.fuse = nn.Sequential(
            ConvBNAct(dim * 2, dim * 2, k=1, p=0, act="SiLU"),
            ConvBNAct(dim * 2, dim * 2, k=3, act="SiLU"),
        )

        # Prediction head for current level Pi.
        self.pred = nn.Sequential(
            ConvBNAct(dim * 2, dim, k=3, act="SiLU"),
            nn.Conv2d(dim, num_classes, kernel_size=1),
        )

    def forward(self, rgb, ir, prev_pair=None, pred_size=None):
        # 修改
        # 1. 跨层融合：深层特征上采样后，与当前层同模态特征拼接
        if prev_pair is not None:
            prev_rgb, prev_ir = prev_pair

            prev_rgb = F.interpolate(prev_rgb, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            prev_ir = F.interpolate(prev_ir, size=ir.shape[-2:], mode="bilinear", align_corners=False)

            rgb = self.rgb_prev_fuse(torch.cat([rgb, prev_rgb], dim=1))
            ir = self.ir_prev_fuse(torch.cat([ir, prev_ir], dim=1))

        diff = rgb - ir

        g_rgb, g_ir = self.diff_gate(diff).chunk(2, dim=1)

        rgb_ms = self.rgb_ms(rgb)
        ir_ms = self.ir_ms(ir)

        rgb = rgb + rgb_ms * g_rgb
        ir = ir + ir_ms * g_ir

        joint = torch.cat([rgb, ir], dim=1)     # [B, 2C, H, W]

        joint = self.fuse(joint)                # [B, 2C, H, W]

        # pred = self.pred(joint)                 # [B, num_classes, H, W]

        # 修改
        # 当前层预测
        if pred_size is not None:
            pred_input = F.interpolate(
                joint,
                size=pred_size,
                mode="bilinear",
                align_corners=False
            )
        else:
            pred_input = joint

        pred = self.pred(pred_input)

        next_rgb, next_ir = joint.chunk(2, dim=1)

        return next_rgb, next_ir, pred

class SFMambaRGBT(nn.Module):
    def __init__(
        self,
        backbone: str = "pvt_v2_b4",
        pretrained_path: Optional[str] = None,
        ir_channels: int = 1,
        num_classes: int = 1,
        freq_depths: Sequence[int] = (1, 1, 1, 1),
        ssm_d_state: int = 16,
        sigma_ratio: float = 0.18,
        drop_path_rate: float = 0.0,
        deep_supervision: bool = True,
        decoder_dim: int = 64,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        # self.spatial_encoder, dims = build_backbone(backbone, pretrained_path=pretrained_path, in_chans=3)
        # self.dims = list(dims)
        # self.spatial_encoder, backbone_dims = build_backbone(
        #     backbone,
        #     pretrained_path=pretrained_path,
        #     in_chans=3
        # )
        self.rgb_encoder, backbone_dims = build_backbone(
            backbone,
            pretrained_path=pretrained_path,
            in_chans=3
        )
        
        self.ir_encoder, ir_backbone_dims = build_backbone(
            backbone,
            pretrained_path=pretrained_path,
            in_chans=3
        )
        
        self.backbone_dims = list(backbone_dims)
        
        # 小通道，先统一 96
        feat_dims = [decoder_dim, decoder_dim, decoder_dim, decoder_dim]
        self.dims = feat_dims
        self.rgb_spatial_proj = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(in_dim, decoder_dim, k=1, p=0),
                ConvBNAct(decoder_dim, decoder_dim, k=3),
            )
            for in_dim in backbone_dims
        ])
        self.ir_spatial_proj = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(in_dim, decoder_dim, k=1, p=0),
                ConvBNAct(decoder_dim, decoder_dim, k=3),
            )
            for in_dim in backbone_dims
        ])

        self.frequency_encoder = FrequencyEncoder(
            in_ch=3,
            dims=feat_dims,
            depths=freq_depths,
            ssm_d_state=ssm_d_state,
            drop_path_rate=drop_path_rate,
        )
        
        self.rgb_band_to_space = nn.ModuleList([
            FrequencyBandToSpace(dim, sigma_ratio=sigma_ratio) for dim in feat_dims
        ])
        self.ir_band_to_space = nn.ModuleList([
            FrequencyBandToSpace(dim, sigma_ratio=sigma_ratio) for dim in feat_dims
        ])
        
        self.cmcd = nn.ModuleList([
            CrossModalCrossDomainFigC(dim, d_state=ssm_d_state) for dim in feat_dims
        ])
        
        self.decoder = nn.ModuleList([
            DifferenceGuidedCrossLevel(
                dim=decoder_dim,
                prev_dim=decoder_dim if i < 3 else None,
                num_classes=num_classes,
            )
            for i in range(4)
        ])
        # 新加11111111111111
        mid_dim = max(decoder_dim // 2, 16)

        self.rgb_pred_heads = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(decoder_dim, mid_dim, k=3),
                nn.Conv2d(mid_dim, num_classes, kernel_size=1),
            )
            for _ in range(4)
        ])
        
        self.ir_pred_heads = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(decoder_dim, mid_dim, k=3),
                nn.Conv2d(mid_dim, num_classes, kernel_size=1),
            )
            for _ in range(4)
        ])



        # self.final_head = nn.Sequential(
        #     ConvBNAct(num_classes * 4, 16, k=3),
        #     nn.Conv2d(16, num_classes, 1),
        # )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, return_dict: bool = False):
        input_size = rgb.shape[-2:]

        # # Shared spatial encoder.
        # rgb_s = self.spatial_encoder(rgb)   # [C1,C2,C3,C4]
        # ir_s = self.spatial_encoder(ir)
        # rgb_s = [self.rgb_spatial_proj[i](rgb_s[i]) for i in range(4)]
        # ir_s = [self.ir_spatial_proj[i](ir_s[i]) for i in range(4)]
        
        # Separate spatial encoders.
        rgb_s = self.rgb_encoder(rgb)       # [C1,C2,C3,C4]
        ir_s = self.ir_encoder(ir)         # [C1,C2,C3,C4]
        
        rgb_s = [
            self.rgb_spatial_proj[i](rgb_s[i])
            for i in range(4)
        ]
        
        ir_s = [
            self.ir_spatial_proj[i](ir_s[i])
            for i in range(4)
        ]

        # Shared frequency encoder.
        rgb_f, rgb_phase = self.frequency_encoder(rgb, rgb_s)
        ir_f, ir_phase = self.frequency_encoder(ir, ir_s)

        # Frequency bands to spatial low/high features.
        rgb_lh = [self.rgb_band_to_space[i](rgb_f[i], rgb_phase[i]) for i in range(4)]
        ir_lh = [self.ir_band_to_space[i](ir_f[i], ir_phase[i]) for i in range(4)]

        # Cross-modal cross-domain fusion.
        rgb_sf, ir_sf = [], []
        for i in range(4):
            rgb_low, rgb_high = rgb_lh[i]
            ir_low, ir_high = ir_lh[i]
            r, t = self.cmcd[i](rgb_s[i], ir_s[i], rgb_low, rgb_high, ir_low, ir_high)
            rgb_sf.append(r)
            ir_sf.append(t)

        native_preds: List[Optional[torch.Tensor]] = [None, None, None, None]
        rgb_native_preds: List[Optional[torch.Tensor]] = [None, None, None, None]
        ir_native_preds: List[Optional[torch.Tensor]] = [None, None, None, None]
        
        prev_pair = None
        
        for i in reversed(range(4)):
            pred_size = input_size if i == 0 else None
        
            next_rgb, next_ir, pred = self.decoder[i](
                rgb_sf[i],
                ir_sf[i],
                prev_pair,
                pred_size=pred_size,
            )
        
            native_preds[i] = pred
        
            # 这里就是用上一阶段/当前 decoder 阶段出来的两个单模态特征
            rgb_native_preds[i] = self.rgb_pred_heads[i](next_rgb)
            ir_native_preds[i] = self.ir_pred_heads[i](next_ir)
        
            prev_pair = (next_rgb, next_ir)

        # # 最终预测直接使用 P1
        # final = side_preds[0]
        side_preds = [
            p if p.shape[-2:] == input_size else F.interpolate(
                p,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )
            for p in native_preds
        ]
        
        rgb_side_preds = [
            p if p.shape[-2:] == input_size else F.interpolate(
                p,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )
            for p in rgb_native_preds
        ]
        
        ir_side_preds = [
            p if p.shape[-2:] == input_size else F.interpolate(
                p,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )
            for p in ir_native_preds
        ]
        
        final = side_preds[0]

        if return_dict:
            return {
                "pred": final,
                "preds": [final] + side_preds[1:],
                "side_preds": side_preds,
                "native_preds": native_preds,
        
                "rgb_preds": rgb_side_preds,
                "ir_preds": ir_side_preds,
                "rgb_native_preds": rgb_native_preds,
                "ir_native_preds": ir_native_preds,
        
                "rgb_spatial": rgb_s,
                "ir_spatial": ir_s,
                "rgb_frequency": rgb_f,
                "ir_frequency": ir_f,
            }


        if self.deep_supervision:
            return [final] + side_preds[1:], rgb_side_preds, ir_side_preds
        
        return final, rgb_side_preds[:1], ir_side_preds[:1]



def build_model(**kwargs):
    return SFMambaRGBT(**kwargs)
