"""Step 4 - train the baseline U-Net (from scratch) on DeepGlobe farmland masks.

Run on Kaggle (GPU T4 x2):
    !mkdir -p /kaggle/working/src
    # paste model.py and dataset.py and train.py into /kaggle/working/src  (%%writefile)
    !python /kaggle/working/src/train.py

Logs: one checkpoint per epoch + best model + runs.csv + a loss/IoU curve PNG,
all in /kaggle/working/outputs/. Kaggle can disconnect, so every epoch is saved.
"""

import csv
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no screen on the server -> write the figure to a file
import matplotlib.pyplot as plt
import numpy as np
import torch                     # PyTorch
import torch.nn as nn            # loss functions
from torch.utils.data import DataLoader   # batches the tiles

from dataset import (            # our own Step 3 code
    INPUT_ROOT, OUT_DIR, find_pairs, load_class_dict, agriculture_rgb,
    split_pairs, build_tiles, DeepGlobeTiles,
)
from model import UNetSmall
from losses import build_loss       # cross-entropy, Dice, or both (Step 7)
from augment import AugDataset      # random flips/rot90, used only for '*_aug' runs

# --------------------------------------------------------------------------- #
# Run settings (keep these IDENTICAL in Step 5 so the comparison is fair)
# --------------------------------------------------------------------------- #
SEED = 42
TRAIN_IMAGES = 120               # subset of train images (81 tiles each)
VAL_IMAGES = 25                  # subset of val images (monitoring only)
BATCH = 16
EPOCHS = 15
LR = 2e-4
NUM_WORKERS = 2                  # CPU processes that load tiles in parallel
# checkpoint names come from the model name (see run_name in main)


def set_seed(seed: int) -> None:
    """Same seed -> same shuffle and same split, so runs are comparable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # seeds the CPU random generator
    torch.cuda.manual_seed_all(seed)  # seeds every GPU


def seg_metrics(pred: torch.Tensor, target: torch.Tensor):
    """IoU / F1 / accuracy for class 1 (farmland), from a batch of predictions."""
    p = pred.reshape(-1) == 1        # predicted farmland, flattened
    t = target.reshape(-1) == 1      # true farmland, flattened
    tp = int((p & t).sum())          # predicted farm AND truly farm
    fp = int((p & ~t).sum())         # predicted farm but not farm (false alarm)
    fn = int((~p & t).sum())         # farm missed
    tn = int((~p & ~t).sum())
    return tp, fp, fn, tn


def metric_dict(tp, fp, fn, tn) -> dict:
    total = tp + fp + fn + tn
    iou = tp / max(tp + fp + fn, 1)          # union = predicted OR true
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    oa = (tp + tn) / max(total, 1)           # overall accuracy (misleading alone)
    pred_share = (tp + fp) / max(total, 1)   # how much farmland the model predicts
    return {"iou": iou, "f1": f1, "oa": oa, "pred_share": pred_share}


def build_model(name: str) -> nn.Module:
    """'scratch' = our own UNetSmall. 'resnet34' = U-Net with an ImageNet-pretrained encoder."""
    if name == "scratch":
        return UNetSmall(n_classes=2)
    if name == "resnet34":
        import segmentation_models_pytorch as smp   # only imported when this option is used
        return smp.Unet(
            encoder_name="resnet34",      # the encoder half of the U-Net
            encoder_weights="imagenet",   # pretrained weights = the transfer-learning part
            in_channels=3,                # RGB images
            classes=2,                    # background + farmland
        )
    raise ValueError(f"unknown model '{name}' - use 'scratch' or 'resnet34'")


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if there is one
    print("device:", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    torch.backends.cudnn.benchmark = True    # lets cuDNN pick the fastest conv kernels

    # ---- data: reuse the Step 3 pipeline exactly -------------------------- #
    pairs = find_pairs(INPUT_ROOT)
    agri = agriculture_rgb(load_class_dict(INPUT_ROOT))
    splits = split_pairs(pairs, SEED)
    train_tiles = build_tiles(splits["train"], agri, limit=TRAIN_IMAGES)
    val_tiles = build_tiles(splits["val"], agri, limit=VAL_IMAGES)
    print(f"tiles: train {len(train_tiles)} | val {len(val_tiles)}")

    use_aug = len(sys.argv) > 2 and sys.argv[2].endswith("_aug")   # 'ce_aug' -> augment
    train_ds = DeepGlobeTiles(train_tiles)        # plain tiles
    if use_aug:                                   # random flips/rot90 on TRAIN tiles only
        train_ds = AugDataset(train_ds, seed=SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(DeepGlobeTiles(val_tiles), batch_size=BATCH, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    # shuffle=True: new tile order each epoch. pin_memory: faster CPU->GPU copy.

    # ---- model, loss, optimiser ------------------------------------------ #
    model_name = sys.argv[1] if len(sys.argv) > 1 else "scratch"   # default = Step 4 baseline
    loss_name = sys.argv[2] if len(sys.argv) > 2 else "ce"        # 'ce', 'dice' or 'ce_dice'
    run_name = f"unet_{model_name}_{loss_name}"   # separate checkpoint names per model/loss
    model = build_model(model_name).to(device)    # move all weights to the GPU
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = build_loss(loss_name.replace("_aug", ""))   # 'ce_aug' uses plain cross-entropy
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")  # AMP helper
    print(f"model: {model_name} | params {n_params:,} | lr {LR} | batch {BATCH} | epochs {EPOCHS}")
    print(f"train {TRAIN_IMAGES} images | val {VAL_IMAGES} images | loss: {loss_name}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}
    best_iou = -1.0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()                    # training mode: BatchNorm uses batch statistics
        running, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)   # image batch -> GPU
            y = y.to(device, non_blocking=True)   # mask batch -> GPU
            opt.zero_grad(set_to_none=True)       # clear old gradients
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)                 # forward pass, in fp16 where safe
                loss = loss_fn(logits, y)         # cross-entropy
            scaler.scale(loss).backward()         # backward pass, scaled for fp16
            scaler.step(opt)                      # update weights
            scaler.update()                       # adjust the scale for next step
            running += loss.item() * x.size(0)    # loss is a batch mean -> weight by batch size
            seen += x.size(0)

        train_loss = running / max(seen, 1)

        # ---- validation: no gradients, no weight updates ------------------
        model.eval()                     # eval mode: BatchNorm uses running statistics
        val_loss_sum, seen_val, tp, fp, fn, tn = 0.0, 0, 0, 0, 0, 0
        with torch.no_grad():            # do not build a graph -> saves memory
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                val_loss_sum += loss.item() * x.size(0)
                seen_val += x.size(0)
                a, b, c, d = seg_metrics(logits.argmax(1), y)  # argmax -> class 0 or 1
                tp, fp, fn, tn = tp + a, fp + b, fn + c, tn + d

        val_loss = val_loss_sum / max(seen_val, 1)
        m = metric_dict(tp, fp, fn, tn)

        for k, v in (("epoch", epoch), ("train_loss", train_loss),
                     ("val_loss", val_loss), ("val_iou", m["iou"]), ("val_f1", m["f1"])):
            history[k].append(v)

        print(f"epoch {epoch:02d}/{EPOCHS} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val IoU {m['iou']:.4f} | F1 {m['f1']:.4f} | OA {m['oa']:.4f} "
              f"| pred_farm {100 * m['pred_share']:.1f}% | {time.time() - t_start:.0f}s", flush=True)

        # save ONLY the best-so-far model (highest val IoU), overwritten each improvement
        if m["iou"] > best_iou:
            best_iou = m["iou"]
            ckpt = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "val_iou": m["iou"], "config": {"lr": LR, "batch": BATCH, "loss": loss_name}}
            torch.save(ckpt, OUT_DIR / f"{run_name}_best.pth")

    total_min = (time.time() - t_start) / 60

    # ---- curve ------------------------------------------------------------ #
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["epoch"], history["train_loss"], label="train")
    ax[0].plot(history["epoch"], history["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy loss")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(history["epoch"], history["val_iou"], marker="o", color="tab:green")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val IoU (farmland)")
    ax[1].set_title("Validation IoU"); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{run_name}_curves.png", dpi=120)
    plt.close(fig)
    print(f"curves -> {OUT_DIR / (run_name + '_curves.png')}")

    # ---- run log (feeds the results table later) -------------------------- #
    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "model": model_name,
           "loss": loss_name, "lr": LR, "batch": BATCH, "epochs": EPOCHS,
           "train_images": TRAIN_IMAGES, "val_images": VAL_IMAGES,
           "final_train_loss": round(history["train_loss"][-1], 4),
           "final_val_loss": round(history["val_loss"][-1], 4),
           "best_val_iou": round(best_iou, 4),
           "val_f1_at_best": round(max(history["val_f1"]), 4),
           "test_iou": "", "minutes": round(total_min, 1)}
    log_path = OUT_DIR / "runs.csv"
    new_file = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)
    print("run logged ->", log_path)

    print(f"\nBEST val IoU {best_iou:.4f} | total time {total_min:.1f} min")
    if best_iou < 0.5:
        print(">>> WARNING: val IoU below 0.5 - stop and report, do not tune blindly.")


if __name__ == "__main__":
    main()
