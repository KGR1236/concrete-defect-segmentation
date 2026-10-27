"""손실함수

- ce      : 클래스 가중 CrossEntropy (배경 0.1, 균열 3, 박리 1)
- ce_dice : 0.5 * CE + 0.5 * Dice
"""
import torch
import torch.nn as nn

CLASS_WEIGHTS = [0.1, 3.0, 1.0]
LOSS_NAMES = ["ce", "ce_dice"]


def dice_loss(pred, target, num_classes: int = 3, smooth: float = 1e-6):
    pred = torch.softmax(pred, dim=1)
    loss = 0
    for cls in range(num_classes):
        pred_c = pred[:, cls]
        target_c = (target == cls).float()
        inter = (pred_c * target_c).sum()
        dice = (2 * inter + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        loss += 1 - dice
    return loss / num_classes


class SegLoss(nn.Module):
    def __init__(self, kind: str = "ce", num_classes: int = 3, weights=CLASS_WEIGHTS):
        super().__init__()
        if kind not in LOSS_NAMES:
            raise ValueError(f"지원하지 않는 손실함수: {kind} (가능: {LOSS_NAMES})")
        self.kind = kind
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    def forward(self, outputs, masks):
        ce = self.ce(outputs, masks)
        if self.kind == "ce":
            return ce
        return 0.5 * ce + 0.5 * dice_loss(outputs, masks, self.num_classes)
