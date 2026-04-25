"""
Training script for ViT-UNet breast cancer model.

Example:
    python -m model.train --data_root data/BUSI --epochs 30 --batch_size 8

If --data_root is empty/missing the dataset class transparently switches
to synthetic data so you can confirm the pipeline runs end-to-end.

Recommended public datasets (after downloading & arranging into the
benign/ malignant/ folder layout described in dataset.py):
    - BUSI: Breast Ultrasound Images Dataset
    - CBIS-DDSM (mammography, requires conversion)
"""
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .vit_unet import ViTUNet, MultiTaskLoss
from .dataset import BreastCancerDataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def dice_score(logits, target, eps=1e-6):
    probs = (torch.sigmoid(logits) > 0.5).float()
    target = target.float()
    inter = (probs * target).sum((1, 2, 3))
    union = probs.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()


def classification_accuracy(logits, target):
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


# ---------------------------------------------------------------------------
# Train / Eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    losses, dices, accs = [], [], []
    for imgs, masks, labels in loader:
        imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss, parts = criterion(out, masks, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss, parts = criterion(out, masks, labels)
            loss.backward()
            optimizer.step()

        losses.append(parts["total"])
        dices.append(dice_score(out["seg_logits"].detach(), masks))
        accs.append(classification_accuracy(out["cls_logits"].detach(), labels))

    return {
        "loss": sum(losses) / len(losses),
        "dice": sum(dices) / len(dices),
        "acc": sum(accs) / len(accs),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, dices, accs = [], [], []
    for imgs, masks, labels in loader:
        imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
        out = model(imgs)
        loss, parts = criterion(out, masks, labels)
        losses.append(parts["total"])
        dices.append(dice_score(out["seg_logits"], masks))
        accs.append(classification_accuracy(out["cls_logits"], labels))

    return {
        "loss": sum(losses) / len(losses),
        "dice": sum(dices) / len(dices),
        "acc": sum(accs) / len(accs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data/BUSI")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", help="Use mixed precision")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device = {device}")

    # ------ Data
    full_ds = BreastCancerDataset(args.data_root, img_size=args.img_size, train=True)
    n_val = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.train = False  # disable aug on val (shared object — best-effort)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"[Train] train={len(train_ds)}  val={len(val_ds)}")

    # ------ Model
    model = ViTUNet(
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Train] model params = {n_params:.1f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = MultiTaskLoss(w_seg=1.0, w_cls=1.0)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_score = -1.0

    # ------ Loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        va = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"[E{epoch:03d}] "
            f"train loss={tr['loss']:.4f} dice={tr['dice']:.3f} acc={tr['acc']:.3f} | "
            f"val loss={va['loss']:.4f} dice={va['dice']:.3f} acc={va['acc']:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{time.time()-t0:.1f}s"
        )

        score = 0.5 * va["dice"] + 0.5 * va["acc"]
        if score > best_score:
            best_score = score
            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val": va,
            }
            torch.save(ckpt, Path(args.checkpoint_dir) / "best.pt")
            print(f"        ↳ saved best (score={score:.4f})")

    # always save last
    torch.save(
        {"model": model.state_dict(), "args": vars(args)},
        Path(args.checkpoint_dir) / "last.pt",
    )
    print(f"[Train] done. Best val score = {best_score:.4f}")


if __name__ == "__main__":
    main()
