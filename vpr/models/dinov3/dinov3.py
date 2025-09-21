import os

import torch
import torch.nn as nn
from . import urls

DINOV3_REPO_PATH = "/home/emmanuel/Desktop/crocodl_challenge/experiments/dino/dinov3"


if not os.path.exists(DINOV3_REPO_PATH):
    raise RuntimeError("Need to include DINOv3 repo")


class DinoV3(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        if not model_name in urls.__dict__:
            raise RuntimeError(f"{model_name} does not exist or is not included")

        self.model = torch.hub.load(
            DINOV3_REPO_PATH,
            model_name,
            source = 'local',
            weights = urls.__dict__[model_name]
        )
    
    def forward(self, x: torch):
        return self.model(x)


    @classmethod
    def ViTBase(cls):
        return cls("dinov3_vitb16")

    @classmethod
    def ViTSmallPlus(cls):
        return cls("dinov3_vitsplus")

    @classmethod
    def ViTLarge(cls):
        return cls("dinov3_vitl16")

    @classmethod
    def ConvexNetLarge(cls):
        return cls("dinov3_convnext_large")
