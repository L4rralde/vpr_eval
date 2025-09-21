import torch


class DinoV2(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        self.model = torch.hub.load(
            'facebookresearch/dinov2',
            model_name
        )

    def forward(self, x: torch):
        return self.model(x)


    @classmethod
    def ViTBase(cls):
        return cls("dinov2_vitb14")

    @classmethod
    def ViTLarge(cls):
        return cls("dinov2_vitl14")

    @classmethod
    def ViTGiant(cls):
        return cls("dinov2_vitg14")
