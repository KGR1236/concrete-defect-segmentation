"""데이터 점검 (concrete_image_check.ipynb 정리본)

1. 이미지-마스크 크기 매칭 확인
2. 마스크 픽셀값 / JSON 라벨 종류 확인 (--json-dir 지정 시)
3. 클래스별 군집(연결 영역) 수
4. 분할(7:1:2)별 균열/박리 픽셀 수

사용 예:
    python -m scripts.check_data --image-dir ... --mask-dir ... [--json-dir ...]
    python -m scripts.check_data --image-dir ... --mask-dir ... --show 5
"""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image
from scipy import ndimage

from src.data.split import list_pairs, stratified_split

# 원본 마스크 기준 클래스 정의
CLASS_MAP = {"균열": [1], "박리": [2], "철근노출": [3, 4, 5]}


def check_sizes(image_dir, mask_dir, pairs):
    bad = 0
    for i, (img_f, mask_f) in enumerate(pairs):
        img_size = Image.open(os.path.join(image_dir, img_f)).size
        mask_size = Image.open(os.path.join(mask_dir, mask_f)).size
        if img_size != mask_size:
            bad += 1
            print(f"[{i + 1:03d}] 크기 불일치: {img_f} {img_size} | {mask_f} {mask_size}")
    print(f"이미지 수: {len(pairs)} / 크기 불일치: {bad}")


def check_values(mask_dir, json_dir=None):
    pixel_values = set()
    for f in sorted(os.listdir(mask_dir)):
        pixel_values.update(np.unique(np.array(Image.open(os.path.join(mask_dir, f)))).tolist())
    print(f"마스크 내 픽셀값 종류: {sorted(pixel_values)}")

    if json_dir:
        labels = set()
        for f in sorted(os.listdir(json_dir)):
            with open(os.path.join(json_dir, f), encoding="utf-8") as fp:
                labels.update(s["label"] for s in json.load(fp)["shapes"])
        print(f"JSON 라벨 목록: {sorted(labels)}")


def count_clusters(mask_dir):
    total = {k: 0 for k in CLASS_MAP}
    for f in sorted(os.listdir(mask_dir)):
        mask = np.array(Image.open(os.path.join(mask_dir, f)))
        for name, values in CLASS_MAP.items():
            _, n = ndimage.label(np.isin(mask, values).astype(np.uint8))
            total[name] += n
    for name, n in total.items():
        print(f"{name:6} 군집 수: {n}")


def count_pixels(mask_dir, mask_list):
    crack = spalling = 0
    for f in mask_list:
        mask = np.array(Image.open(os.path.join(mask_dir, f)))
        crack += int((mask == 1).sum())
        spalling += int(np.isin(mask, [2, 3, 4, 5]).sum())
    return crack, spalling


def show_samples(image_dir, mask_dir, pairs, n):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cmap = plt.get_cmap("tab10", 10)
    for img_f, mask_f in random.sample(pairs, min(n, len(pairs))):
        img = Image.open(os.path.join(image_dir, img_f))
        mask = np.array(Image.open(os.path.join(mask_dir, mask_f)))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title(f"Image: {img_f}")
        axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=9)
        axes[1].set_title(f"Mask: {mask_f}")
        axes[1].legend(
            handles=[Patch(color=cmap(v), label=f"Pixel value {v}") for v in np.unique(mask)],
            loc="lower right", fontsize=9,
        )
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


def main():
    p = argparse.ArgumentParser(description="콘크리트 데이터 점검")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--json-dir", default=None)
    p.add_argument("--show", type=int, default=0, help="무작위 샘플 시각화 개수")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pairs = list_pairs(args.image_dir, args.mask_dir)

    print("===== 1. 크기 매칭 =====")
    check_sizes(args.image_dir, args.mask_dir, pairs)
    print("\n===== 2. 픽셀값 / 라벨 =====")
    check_values(args.mask_dir, args.json_dir)
    print("\n===== 3. 군집 수 =====")
    count_clusters(args.mask_dir)
    print("\n===== 4. 분할 =====")
    splits = stratified_split(args.image_dir, args.mask_dir, seed=args.seed)
    print(f"\n{'':8} {'균열 픽셀':>15} {'박리 픽셀':>15}")
    for name in ["train", "val", "test"]:
        crack, spall = count_pixels(args.mask_dir, splits[name][1])
        print(f"{name:8} {crack:>15,} {spall:>15,}")

    if args.show > 0:
        show_samples(args.image_dir, args.mask_dir, pairs, args.show)


if __name__ == "__main__":
    main()
