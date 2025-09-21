import sys
import os

import torch

sys.path.append('/home/emmanuel/Desktop/crocodl_challenge/experiments/dino/salad')
from vpr_model import VPRModel


def load_model(ckpt_path):
    model = VPRModel(
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
    )

    model.load_state_dict(torch.load(ckpt_path))
    model = model.eval()
    model = model.to('cuda')
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model


class DinoSalad(torch.nn.Module):
    ckpt_path = '/home/emmanuel/Desktop/crocodl_challenge/experiments/dino/salad_results/weights/dino_salad.ckpt'
    def __init__(self):
        super().__init__()
        if not os.path.exists(DinoSalad.ckpt_path):
            RuntimeError("Check the weights exist")
        self.model = load_model(DinoSalad.ckpt_path).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
