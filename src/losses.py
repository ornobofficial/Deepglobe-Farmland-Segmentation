"""Loss functions for the binary farmland segmentation.

Cross-entropy looks at every pixel independently.
Dice looks at the overlap of the whole mask, so it pushes back when farmland
is a small part of the image - that is the "innovation" we compare in Step 7.
"""

import torch                 # tensors
import torch.nn as nn        # base class for losses


class DiceLoss(nn.Module):
    """1 - Dice coefficient, computed on the farmland class only."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps       # avoids dividing by zero for empty masks

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, 2, H, W) raw scores; target: (B, H, W) class ids 0/1
        prob = torch.softmax(logits, dim=1)[:, 1]     # farmland probability per pixel
        t = target.float()                            # 0.0 / 1.0 mask
        inter = (prob * t).sum(dim=(1, 2))            # soft overlap, per image in the batch
        union = prob.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()                        # loss = 1 - mean Dice


class CombinedLoss(nn.Module):
    """Cross-entropy + Dice. The plain sum, so both get equal weight."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()               # per-pixel term
        self.dice = DiceLoss()                        # region-overlap term

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice(logits, target)


def build_loss(name: str) -> nn.Module:
    """Pick a loss by name: 'ce', 'dice' or 'ce_dice'."""
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "dice":
        return DiceLoss()
    if name == "ce_dice":
        return CombinedLoss()
    raise ValueError(f"unknown loss '{name}' - use 'ce', 'dice' or 'ce_dice'")


if __name__ == "__main__":
    # sanity check on a tiny fake batch: logits strongly favour the true class
    logits = torch.tensor([[[[9.0, -9.0]], [[-9.0, 9.0]]]])   # (B=1, C=2, H=1, W=2)
    target = torch.tensor([[[0, 1]]])                          # (B=1, H=1, W=2) class ids
    for name in ("ce", "dice", "ce_dice"):
        print(name, float(build_loss(name)(logits, target)))   # all three should be ~0
