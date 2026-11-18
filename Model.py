import torch
import torch.nn as nn

# from models.sfmamba_rgbt import SFMambaRGBT
# from models.sfmamba_rgbt_v2back import SFMambaRGBT
# from models.sfmamba_rgbt_v3 import SFMambaRGBT
from models.sfmamba_rgbt_v3_2backbone import SFMambaRGBT

class Network(nn.Module):
    def __init__(
        self,
        channels=96,
        backbone="resnet101d",
        pretrained_path=None,
        num_classes=1,
    ):
        super().__init__()

        self.net = SFMambaRGBT(
            backbone=backbone,
            pretrained_path=pretrained_path,
            num_classes=num_classes,
            decoder_dim=channels,
            deep_supervision=True,
        )

    def forward(self, rgb, ir):
        out = self.net(rgb, ir, return_dict=True)
    
        preds = out["preds"]
        rgb_preds = out["rgb_preds"]
        ir_preds = out["ir_preds"]
    
        while len(preds) < 4:
            preds.append(preds[-1])
    
        while len(rgb_preds) < 4:
            rgb_preds.append(rgb_preds[-1])
    
        while len(ir_preds) < 4:
            ir_preds.append(ir_preds[-1])
    
        return preds[:4], rgb_preds[:4], ir_preds[:4]

