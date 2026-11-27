"""Step 6 - evaluate the best model ONCE on the held-out test split.

Run on Kaggle:
    !python /kaggle/working/src/evaluate.py

The test split (121 images) was never used for training or tuning.
This script prints IoU / F1 / OA, the predicted-farmland share, and saves
a figure with image | true mask | predicted mask for 6 tiles.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # write the figure to a file (no screen)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch                     # PyTorch
import segmentation_models_pytorch as smp   # same architecture as the trained model

from dataset import (            # our Step 3 code: discovery, colours, split, binary mask
    INPUT_ROOT, OUT_DIR, TILE, find_pairs, load_class_dict, agriculture_rgb,
    split_pairs, to_binary,
)
from model import UNetSmall      # the from-scratch U-Net, needed for 'scratch' checkpoints

CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR / "unet_resnet34_ce_best.pth"   # best val IoU = 0.7237 (epoch 14)
BATCH = 16                                     # tiles per forward pass
N_FIG = 6                                      # tiles shown in the figure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_rgb_and_mask(img_path: Path, mask_path: Path, agri):
    """Read one image + its colour mask -> (HxWx3 uint8 image, HxW uint8 binary mask)."""
    img = np.asarray(Image.open(img_path).convert("RGB"))
    msk = np.asarray(Image.open(mask_path).convert("RGB"))
    if msk.shape[:2] != img.shape[:2]:
        msk = np.asarray(
            Image.fromarray(msk).resize((img.shape[1], img.shape[0]), Image.NEAREST)
        )
    return img, to_binary(msk, agri)


def cut_tiles(img, mask, tile: int = TILE):
    """Yield non-overlapping tile x tile pieces, same rule as training."""
    h, w = img.shape[:2]
    for r in range(0, h - tile + 1, tile):
        for c in range(0, w - tile + 1, tile):
            yield img[r:r + tile, c:c + tile], mask[r:r + tile, c:c + tile]


def parse_ckpt_name(stem: str) -> tuple[str, str, int]:
    """'unet_resnet34_ce_aug_s43_best' -> ('resnet34', 'ce_aug', 43). No hard-coded names."""
    body = stem.removeprefix("unet_").removesuffix("_best")
    seed = 42                                   # default, for checkpoints without a seed suffix
    head, sep, tail = body.rpartition("_s")
    if sep and tail.isdigit():                  # "..._s43" -> seed 43
        body, seed = head, int(tail)
    for loss in ("ce_dice", "ce_aug", "dice", "ce"):   # longest first, so ce_aug beats ce
        if body.endswith("_" + loss):
            return body[: -len(loss) - 1], loss, seed
    return body, "unknown", seed


def record_test_iou(model: str, loss: str, seed: int, val_iou: float, test_iou: float) -> None:
    """Write test IoU back into the runs.csv row that produced this checkpoint."""
    path = OUT_DIR / "runs.csv"
    if not path.exists():
        print("runs.csv not found - nothing to update")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    best_i, best_d = None, 1e9
    for i, r in enumerate(rows):
        if r["model"] == model and r["loss"] == loss and str(r.get("seed", "")) == str(seed):
            d = abs(float(r["best_val_iou"]) - val_iou)   # pick the run with this val IoU
            if d < best_d:
                best_i, best_d = i, d
    if best_i is None:
        print(f"no runs.csv row for model={model} loss={loss} - not updated")
        return
    rows[best_i]["test_iou"] = f"{test_iou:.4f}"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"runs.csv updated: test_iou {test_iou:.4f} for {model}/{loss} (val {val_iou:.4f})")


def build_eval_model(model_name: str):
    """Pick the architecture from the checkpoint name: 'scratch' -> UNetSmall, else ResNet34.

    encoder_weights=None on purpose: the checkpoint overwrites every parameter,
    so downloading ImageNet weights here would be pointless (and can fail offline).
    """
    if model_name == "scratch":
        return UNetSmall(n_classes=2)
    return smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=2)


def main() -> None:
    torch.manual_seed(0)                       # fixed seed -> same example tiles each run
    print("device:", device)

    pairs = find_pairs(INPUT_ROOT)
    agri = agriculture_rgb(load_class_dict(INPUT_ROOT))
    test_pairs = split_pairs(pairs, 42)["test"]     # 121 images, never seen before
    print(f"test images: {len(test_pairs)}")

    # Build the architecture WITHOUT downloading weights: our checkpoint overwrites
    # every parameter anyway, so the pretrained download is pointless here.
    ckpt_model, ckpt_loss, ckpt_seed = parse_ckpt_name(CKPT.stem)   # read from the filename
    model = build_eval_model(ckpt_model)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)  # our own file
    model.load_state_dict(ckpt["model"])       # copy the trained weights in
    model.to(device).eval()                    # GPU + eval mode (BatchNorm frozen)
    print(f"loaded {CKPT.name} | model {ckpt_model} | loss {ckpt_loss} | seed {ckpt_seed} | "
          f"epoch {ckpt['epoch']} | val IoU {ckpt['val_iou']:.4f}")

    tp = fp = fn = tn = 0
    shown = 0
    fig, axes = plt.subplots(N_FIG, 3, figsize=(9, 3 * N_FIG))

    for img_path, mask_path in test_pairs:
        img, mask = load_rgb_and_mask(img_path, mask_path, agri)
        tiles = list(cut_tiles(img, mask))

        for start in range(0, len(tiles), BATCH):
            chunk = tiles[start:start + BATCH]
            x = np.stack([t[0] for t in chunk])                 # (B, H, W, 3) uint8
            y = np.stack([t[1] for t in chunk])                 # (B, H, W) uint8
            xb = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0  # -> (B,3,H,W) 0..1
            xb = xb.to(device, non_blocking=True)

            with torch.no_grad():                               # no gradients needed
                with torch.amp.autocast("cuda", dtype=torch.float16,
                                        enabled=device.type == "cuda"):
                    logits = model(xb)                          # forward pass only
            pred = logits.argmax(1).cpu().numpy().astype(np.uint8)   # (B,H,W) 0/1

            p = pred.reshape(-1) == 1
            t = y.reshape(-1) == 1
            tp += int((p & t).sum())
            fp += int((p & ~t).sum())
            fn += int((~p & t).sum())
            tn += int((~p & ~t).sum())

            # collect a few example tiles from DIFFERENT images
            for j in range(len(chunk)):
                frac = float(y[j].mean())
                if shown < N_FIG and 0.05 < frac < 0.95:
                    ax = axes[shown]
                    ax[0].imshow(chunk[j][0]); ax[0].set_title(f"image {img_path.name[:12]}", fontsize=8)
                    ax[1].imshow(chunk[j][1], cmap="gray", vmin=0, vmax=1); ax[1].set_title("true mask", fontsize=8)
                    ax[2].imshow(pred[j], cmap="gray", vmin=0, vmax=1); ax[2].set_title("prediction", fontsize=8)
                    for a in ax:
                        a.axis("off")
                    shown += 1
                    break            # at most one tile per image -> variety

    total = tp + fp + fn + tn
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    oa = (tp + tn) / max(total, 1)
    pred_share = (tp + fp) / max(total, 1)
    true_share = (tp + fn) / max(total, 1)

    print("\n=== TEST SET (121 images, used once) ===")
    print(f"pixels       : {total:,}")
    print(f"IoU (farm)   : {iou:.4f}")
    print(f"F1 / Dice    : {f1:.4f}")
    print(f"overall acc  : {oa:.4f}")
    print(f"pred farmland: {100 * pred_share:.1f}%  (true {100 * true_share:.1f}%)")
    print(f"confusion    : tp {tp:,} | fp {fp:,} | fn {fn:,} | tn {tn:,}")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"test_predictions_{CKPT.stem}.png", dpi=110)
    plt.close(fig)
    print("figure ->", OUT_DIR / f"test_predictions_{CKPT.stem}.png")

    metrics = {"model": ckpt_model, "loss": ckpt_loss, "seed": ckpt_seed, "split": "test",
               "n_images": len(test_pairs), "iou": round(iou, 4), "f1": round(f1, 4),
               "oa": round(oa, 4), "pred_farm_share": round(pred_share, 4),
               "true_farm_share": round(true_share, 4), "checkpoint": CKPT.name}
    (OUT_DIR / f"test_metrics_{CKPT.stem}.json").write_text(json.dumps(metrics, indent=2))
    print("metrics ->", OUT_DIR / f"test_metrics_{CKPT.stem}.json")

    record_test_iou(ckpt_model, ckpt_loss, ckpt_seed, float(ckpt["val_iou"]), iou)  # fill runs.csv

    if iou < 0.5:
        print(">>> WARNING: test IoU below 0.5 - do not tune on test, report as is.")


if __name__ == "__main__":
    main()
