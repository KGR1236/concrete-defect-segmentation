"""콘크리트 결함 분할 데이터셋

원본 마스크 픽셀값: 0=배경, 1=균열, 2=박리, 3/4/5=철근노출
학습에서는 철근노출(3, 4, 5)을 박리(2)로 합쳐 3클래스로 사용한다.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.augment import apply_augment

CLASS_NAMES = ["Background", "Crack", "Spalling"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def remap_mask(mask: torch.Tensor) -> torch.Tensor:
    """철근노출(3, 4, 5) -> 박리(2)"""
    mask[(mask == 3) | (mask == 4) | (mask == 5)] = 2
    return mask


class ConcreteDataset(Dataset):
    def __init__(self, img_list, mask_list, img_dir, mask_dir, image_size: int = 512):
        """
        mask_list 원소는 파일명(str) 또는 (파일명, 증강종류) 튜플.
        """
        self.img_list = img_list
        self.mask_list = mask_list
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_f = self.img_list[idx]
        mask_entry = self.mask_list[idx]
        if isinstance(mask_entry, tuple):
            mask_f, aug_type = mask_entry
        else:
            mask_f, aug_type = mask_entry, "orig"

        size = (self.image_size, self.image_size)
        img = Image.open(os.path.join(self.img_dir, img_f)).convert("RGB").resize(size)
        mask = Image.open(os.path.join(self.mask_dir, mask_f)).resize(size, Image.NEAREST)

        img_np = np.array(img, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)
        img_np, mask_np = apply_augment(img_np, mask_np, aug_type)

        img_np = (img_np.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img_t = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32)
        mask_t = remap_mask(torch.tensor(mask_np.astype(np.int64), dtype=torch.long))
        return img_t, mask_t
