"""학습/검증 루프, 조기 종료, 시드 고정"""
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:
        pass


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        total_loss += criterion(model(imgs), masks).item()
    return total_loss / len(loader)


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        save_path: str, epochs: int = 100, patience: int = 15):
    """val_loss 기준 best 모델 저장 + 조기 종료. (best_epoch, best_val_loss) 반환"""
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f"  → Best model saved (epoch {epoch})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return best_epoch, best_val_loss
