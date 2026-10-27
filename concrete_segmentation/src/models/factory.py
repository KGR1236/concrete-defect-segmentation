"""모델 생성: U-Net / DeepLabV3+ / SegFormer

모든 모델은 forward(x) -> (B, C, H, W) logits 형태로 통일한다.
SegFormer는 1/4 해상도로 출력하므로 입력 크기로 bilinear 업샘플링한다.
"""
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAMES = ["unet", "deeplabv3plus", "segformer"]

# 노트북에서 사용한 모델별 기본 학습률
DEFAULT_LR = {"unet": 1e-4, "deeplabv3plus": 1e-4, "segformer": 1e-5}


class SegformerWrapper(nn.Module):
    def __init__(self, num_classes: int, pretrained: str = "nvidia/mit-b2"):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        config = SegformerConfig.from_pretrained(pretrained, num_labels=num_classes)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            config=config,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(name: str, num_classes: int = 3, encoder: str = "resnet34") -> nn.Module:
    if name == "segformer":
        return SegformerWrapper(num_classes)

    import segmentation_models_pytorch as smp

    if name == "unet":
        arch = smp.Unet
    elif name == "deeplabv3plus":
        arch = smp.DeepLabV3Plus
    else:
        raise ValueError(f"지원하지 않는 모델: {name} (가능: {MODEL_NAMES})")

    return arch(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
