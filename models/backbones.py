from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyBackbone(nn.Module):
    """Only for CPU/debug demo."""
    def __init__(self, in_chans: int = 3, channels=(32, 64, 128, 256)):
        super().__init__()
        self.channels = list(channels)
        layers = []
        prev = in_chans
        for i, ch in enumerate(channels):
            stride = 4 if i == 0 else 2
            layers.append(
                nn.Sequential(
                    nn.Conv2d(prev, ch, 3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.SiLU(inplace=True),
                )
            )
            prev = ch
        self.stages = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs
class ResNetTimmBackbone(nn.Module):
    def __init__(self, name="resnet101d", pretrained_path=None, pretrained=False):
        super().__init__()
        import timm

        self.body = timm.create_model(
            model_name=name,
            features_only=True,
            out_indices=range(1, 5),
            pretrained=pretrained and pretrained_path is None,
        )

        # resnet101d 输出一般是 [256, 512, 1024, 2048]
        self.channels = list(self.body.feature_info.channels())

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")

            if isinstance(ckpt, dict):
                for key in ["model", "state_dict", "net", "params"]:
                    if key in ckpt:
                        ckpt = ckpt[key]
                        break

            clean = {}
            for k, v in ckpt.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                if k.startswith("body."):
                    k = k[len("body."):]
                clean[k] = v

            msg = self.body.load_state_dict(clean, strict=False)
            print(f"[ResNetTimmBackbone] loaded {pretrained_path}: {msg}")

    def forward(self, x):
        outs = self.body(x)
        return list(outs)


class PVTv2Backbone(nn.Module):
    """Adapter for bundled pvtv2_encoder.

    Bundled PVT returns [C4, C3, C2, C1]; this adapter returns [C1, C2, C3, C4].
    """

    dims_map = {
        "pvt_v2_b0": [32, 64, 160, 256],
        "pvt_v2_b1": [64, 128, 320, 512],
        "pvt_v2_b2": [64, 128, 320, 512],
        "pvt_v2_b3": [64, 128, 320, 512],
        "pvt_v2_b4": [64, 128, 320, 512],
        "pvt_v2_b5": [64, 128, 320, 512],
    }

    def __init__(self, name: str = "pvt_v2_b4", pretrained_path: Optional[str] = None):
        super().__init__()
        from . import pvtv2_encoder

        if not hasattr(pvtv2_encoder, name):
            raise ValueError(f"Unknown PVTv2 backbone: {name}")

        self.name = name
        self.body = getattr(pvtv2_encoder, name)()
        self.channels = self.dims_map[name]

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            if isinstance(ckpt, dict):
                for key in ["model", "state_dict"]:
                    if key in ckpt:
                        ckpt = ckpt[key]
                        break
            clean = {}
            for k, v in ckpt.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                clean[k] = v
            msg = self.body.load_state_dict(clean, strict=False)
            print(f"[PVTv2Backbone] loaded {pretrained_path}: {msg}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = list(self.body(x))
        return outs[::-1]


def build_backbone(
    name: str = "pvt_v2_b4",
    pretrained_path: Optional[str] = None,
    in_chans: int = 3,
) -> Tuple[nn.Module, List[int]]:
    name = name.lower()

    if name == "tiny":
        net = TinyBackbone(in_chans=in_chans)
        return net, net.channels

    if name.startswith("pvt_v2"):
        if in_chans != 3:
            raise ValueError("Bundled PVTv2 expects 3-channel input. Use the model's IR adapter for IR.")
        net = PVTv2Backbone(name, pretrained_path=pretrained_path)
        return net, net.channels
    if name in {"resnet101d", "resnet101", "resnet50d", "resnet50"}:
        if in_chans != 3:
            raise ValueError("ResNet backbone expects 3-channel input.")
        net = ResNetTimmBackbone(name=name, pretrained_path=pretrained_path)
        return net, net.channels

    raise ValueError(f"Unsupported backbone: {name}. Use tiny or pvt_v2_b0/b1/b2/b3/b4/b5.")
