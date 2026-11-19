"""VGG16 sliced-optimal-transport style loss from the official texture NCA notebook."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights


STYLE_LAYERS = (1, 6, 11, 18, 25)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGStyleLoss:
    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(self.device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(IMAGENET_STD, device=self.device, dtype=torch.float32)[:, None, None]

    def features(self, imgs: torch.Tensor) -> list[torch.Tensor]:
        x = (imgs - self.mean) / self.std
        b, c, h, w = x.shape
        features = [x.reshape(b, c, h * w)]
        for i, layer in enumerate(self.vgg[: max(STYLE_LAYERS) + 1]):
            x = layer(x)
            if i in STYLE_LAYERS:
                b, c, h, w = x.shape
                features.append(x.reshape(b, c, h * w))
        return features

    @staticmethod
    def project_sort(x: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcn,cp->bpn", x, proj).sort(dim=-1).values

    def ot_loss(self, source: torch.Tensor, target: torch.Tensor, proj_n: int = 32) -> torch.Tensor:
        ch, n = source.shape[-2:]
        projs = F.normalize(torch.randn(ch, proj_n, device=source.device), dim=0)
        source_proj = self.project_sort(source, projs)
        target_proj = self.project_sort(target, projs)
        target_interp = F.interpolate(target_proj, size=n, mode="nearest")
        return (source_proj - target_interp).square().sum()

    def target_features(self, target_img: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            return [f.detach() for f in self.features(target_img)]

    def __call__(self, imgs: torch.Tensor, target_features: list[torch.Tensor]) -> torch.Tensor:
        source = self.features(imgs)
        return sum(self.ot_loss(x, y) for x, y in zip(source, target_features))


def overflow_loss(state: torch.Tensor) -> torch.Tensor:
    return (state - state.clamp(-1.0, 1.0)).abs().sum()


def normalize_gradients(model: torch.nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= p.grad.norm() + 1e-8
