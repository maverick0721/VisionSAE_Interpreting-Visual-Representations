import timm
import torch.nn as nn

class VisionBackbone(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(name, pretrained=pretrained)
        self.model.reset_classifier(0)

    def forward(self, x, layer_idx=None):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x)

        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if layer_idx is not None and i == layer_idx:
                return x.mean(dim=1)

        x = self.model.norm(x)
        return x.mean(dim=1)