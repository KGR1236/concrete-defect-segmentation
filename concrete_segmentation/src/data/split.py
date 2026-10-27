"""이미지/마스크 쌍을 학습:검증:테스트 = 7:1:2 로 분할

박리(픽셀값 2) 포함 이미지가 적기 때문에, 포함/미포함 그룹을 따로 나눈 뒤
합쳐서 각 분할에 박리 이미지가 고르게 들어가도록 한다(층화 분할).
"""
import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

SPALLING_VALUE = 2


def list_pairs(image_dir: str, mask_dir: str):
    """정렬된 파일명 기준으로 (이미지, 마스크) 쌍 목록을 만든다."""
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    if len(image_files) != len(mask_files):
        raise ValueError(
            f"이미지 수({len(image_files)})와 마스크 수({len(mask_files)})가 다릅니다."
        )
    return list(zip(image_files, mask_files))


def has_value(mask_path: str, value: int) -> bool:
    return value in np.unique(np.array(Image.open(mask_path)))


def split_7_1_2(pairs, seed: int = 42):
    imgs = [p[0] for p in pairs]
    masks = [p[1] for p in pairs]
    tr_i, tmp_i, tr_m, tmp_m = train_test_split(imgs, masks, test_size=0.3, random_state=seed)
    va_i, te_i, va_m, te_m = train_test_split(tmp_i, tmp_m, test_size=0.67, random_state=seed)
    return (tr_i, tr_m), (va_i, va_m), (te_i, te_m)


def stratified_split(image_dir: str, mask_dir: str, seed: int = 42, verbose: bool = True):
    """박리 포함 여부로 층화한 7:1:2 분할.

    Returns:
        dict: {"train": (imgs, masks), "val": (imgs, masks), "test": (imgs, masks)}
    """
    has_2, no_2 = [], []
    for img_f, mask_f in list_pairs(image_dir, mask_dir):
        if has_value(os.path.join(mask_dir, mask_f), SPALLING_VALUE):
            has_2.append((img_f, mask_f))
        else:
            no_2.append((img_f, mask_f))

    a = split_7_1_2(has_2, seed)
    b = split_7_1_2(no_2, seed)

    splits = {}
    for i, name in enumerate(["train", "val", "test"]):
        splits[name] = (a[i][0] + b[i][0], a[i][1] + b[i][1])

    if verbose:
        print(f"픽셀값 2 포함: {len(has_2)}장 / 미포함: {len(no_2)}장")
        for i, name in enumerate(["train", "val", "test"]):
            print(f"  {name:5}: {len(splits[name][0]):4d}장 (박리 포함 {len(a[i][0])}장)")

    return splits
