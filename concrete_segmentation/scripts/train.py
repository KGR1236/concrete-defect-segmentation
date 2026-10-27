"""콘크리트 결함(균열/박리) 분할 모델 학습 + 테스트 평가

노트북의 실험 조건을 옵션으로 재현한다.
    - 기본           : --loss ce
    - 손실함수 변경  : --loss ce_dice
    - 단순 증강      : --augment
    - 증강 + 손실변경: --augment --loss ce_dice

사용 예:
    python -m scripts.train --model unet \
        --image-dir /path/to/imagedata_concrete --mask-dir /path/to/maskdata_concrete \
        --augment --loss ce_dice
"""
import argparse
import os

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.augment import expand_with_augments
from src.data.dataset import NUM_CLASSES, ConcreteDataset
from src.data.split import stratified_split
from src.models.factory import DEFAULT_LR, MODEL_NAMES, build_model
from src.training.losses import LOSS_NAMES, SegLoss
from src.training.metrics import compute_iou
from src.training.trainer import fit, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="콘크리트 결함 분할 학습")
    p.add_argument("--model", choices=MODEL_NAMES, required=True)
    p.add_argument("--image-dir", required=True, help="원본 이미지 폴더")
    p.add_argument("--mask-dir", required=True, help="마스크 이미지 폴더")
    p.add_argument("--loss", choices=LOSS_NAMES, default="ce")
    p.add_argument("--augment", action="store_true", help="단순 증강(8배) 사용")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=None, help="미지정 시 모델별 기본값")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    splits = stratified_split(args.image_dir, args.mask_dir, seed=args.seed)
    train_imgs, train_masks = splits["train"]
    if args.augment:
        n = len(train_imgs)
        train_imgs, train_masks = expand_with_augments(train_imgs, train_masks)
        print(f"증강: 학습 이미지 {n}장 → {len(train_imgs)}장")

    def loader(imgs, masks, shuffle):
        ds = ConcreteDataset(imgs, masks, args.image_dir, args.mask_dir, args.image_size)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers)

    train_loader = loader(train_imgs, train_masks, True)
    val_loader = loader(*splits["val"], False)
    test_loader = loader(*splits["test"], False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(args.model, NUM_CLASSES).to(device)
    criterion = SegLoss(args.loss, NUM_CLASSES).to(device)
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.model]
    opt_cls = AdamW if args.model == "segformer" else Adam
    optimizer = opt_cls(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.model}_{'aug' if args.augment else 'noaug'}_{args.loss}"
    save_path = os.path.join(args.out_dir, f"best_{tag}.pth")

    best_epoch, best_val_loss = fit(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        save_path, epochs=args.epochs, patience=args.patience,
    )

    print(f"\nBest epoch: {best_epoch} | Best val_loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    result = compute_iou(model, test_loader, device)
    print("\n===== Test IoU =====")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
