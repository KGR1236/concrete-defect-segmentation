"""단순(전통적) 증강

학습 이미지 1장마다 아래 8가지 버전을 만들어 학습 셋을 8배로 늘린다.
기하 변환(flip/회전)은 마스크에도 똑같이 적용하고, 밝기/노이즈는 이미지에만 적용한다.
"""
import numpy as np

AUGMENT_TYPES = [
    "orig",
    "hflip",
    "vflip",
    "rot90_l",
    "rot90_r",
    "bright_dark",
    "bright_light",
    "noise",
]


def apply_augment(img_np: np.ndarray, mask_np: np.ndarray, aug_type: str):
    """img_np: (H, W, 3) uint8, mask_np: (H, W) uint8"""
    if aug_type == "orig":
        pass
    elif aug_type == "hflip":
        img_np = img_np[:, ::-1, :].copy()
        mask_np = mask_np[:, ::-1].copy()
    elif aug_type == "vflip":
        img_np = img_np[::-1, :, :].copy()
        mask_np = mask_np[::-1, :].copy()
    elif aug_type == "rot90_l":
        img_np = np.rot90(img_np, k=1).copy()
        mask_np = np.rot90(mask_np, k=1).copy()
    elif aug_type == "rot90_r":
        img_np = np.rot90(img_np, k=3).copy()
        mask_np = np.rot90(mask_np, k=3).copy()
    elif aug_type == "bright_dark":
        img_np = np.clip(img_np * 0.7, 0, 255).astype(np.uint8)
    elif aug_type == "bright_light":
        img_np = np.clip(img_np * 1.3, 0, 255).astype(np.uint8)
    elif aug_type == "noise":
        noise = np.random.normal(0, 15, img_np.shape).astype(np.float32)
        img_np = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"알 수 없는 증강 종류: {aug_type}")
    return img_np, mask_np


def expand_with_augments(img_list, mask_list, augment_types=AUGMENT_TYPES):
    """(이미지, 마스크) 목록을 (이미지, (마스크, 증강종류)) 목록으로 확장"""
    aug_imgs, aug_masks = [], []
    for img_f, mask_f in zip(img_list, mask_list):
        for aug in augment_types:
            aug_imgs.append(img_f)
            aug_masks.append((mask_f, aug))
    return aug_imgs, aug_masks
