"""클래스별 IoU / Precision / Recall 및 mIoU(배경 제외)"""
import numpy as np
import torch

from src.data.dataset import CLASS_NAMES


@torch.no_grad()
def compute_iou(model, loader, device, class_names=CLASS_NAMES):
    model.eval()
    n = len(class_names)
    inter = [0] * n
    union = [0] * n
    fp = [0] * n
    fn = [0] * n

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs).argmax(dim=1)
        for cls in range(n):
            p, t = preds == cls, masks == cls
            inter[cls] += (p & t).sum().item()
            union[cls] += (p | t).sum().item()
            fp[cls] += (p & ~t).sum().item()
            fn[cls] += (~p & t).sum().item()

    result = {}
    for cls, name in enumerate(class_names):
        tp = inter[cls]
        result[name] = tp / union[cls] if union[cls] > 0 else 0.0
        result[f"{name}_Precision"] = tp / (tp + fp[cls] + 1e-6)
        result[f"{name}_Recall"] = tp / (tp + fn[cls] + 1e-6)

    result["mIoU"] = float(np.mean([result[name] for name in class_names if name != "Background"]))
    return result
