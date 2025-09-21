import torch
import torch.nn as nn



#During training, images are resized to 224×224, while for inference we resize them to 322×322
#They use imagenet mean_stad

class MegaLoc(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("gmberton/MegaLoc", "get_trained_model").eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)
