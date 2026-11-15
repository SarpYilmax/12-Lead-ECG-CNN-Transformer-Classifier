import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)

        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class HybridLoss(nn.Module):
    def __init__(self, focal_weight: float = 0.7, ce_weight: float = 0.3,
                 gamma: float = 2.0, alpha=None, label_smoothing: float = 0.1):
        super().__init__()
        assert abs(focal_weight + ce_weight - 1.0) < 1e-6, "focal_weight + ce_weight must equal 1.0"
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        fl = self.focal(logits, targets)
        ce = self.ce(logits, targets)
        return self.focal_weight * fl + self.ce_weight * ce
