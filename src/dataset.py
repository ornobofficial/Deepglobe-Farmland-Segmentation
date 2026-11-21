"""Step 3 - DeepGlobe Land Cover -> binary farmland mask -> 256x256 tiles.

Run on Kaggle (after "Add data" -> DeepGlobe Land Cover, by balraj98):

    !python /kaggle/working/src/dataset.py

What it does, in order:
  1. prints the real folder tree under /kaggle/input (paths are DISCOVERED, not assumed)
  2. reads class_dict.csv and picks the "agriculture" RGB colour
  3. builds a binary mask: farmland = 1, everything else = 0
  4. splits by IMAGE 70/15/15 with a fixed seed (never by tile)
  5. cuts non-overlapping 256x256 tiles (small subset first)
  6. saves 4 image/mask samples and prints the overall farmland percentage
"""

import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # script has no screen -> render figures to a file
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch                     # PyTorch: tensors + Dataset
from torch.utils.data import Dataset  # base class our tile dataset inherits from

# --------------------------------------------------------------------------- #
# Settings (change only here)
# --------------------------------------------------------------------------- #
INPUT_ROOT = Path("/kaggle/input")      # where Kaggle mounts "Add data"
OUT_DIR = Path("/kaggle/working/outputs")
TILE = 256                              # tile size in pixels
SEED = 42                               # fixed seed -> same split every run
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15       # rest = test
MAX_IMAGES = {"train": 20, "val": 5, "test": 5}   # small subset first


# --------------------------------------------------------------------------- #
# 1. Discovery
# --------------------------------------------------------------------------- #
def show_tree(root: Path, max_depth: int = 3, files_per_dir: int = 4) -> None:
    """Print the folder tree (dirs + a few file names), so we SEE the real paths."""
    print(f"\n=== 1. folder tree under {root} ===")
    for d in sorted([root] + [p for p in root.rglob("*") if p.is_dir()]):
        depth = len(d.relative_to(root).parts)
        if depth > max_depth:
            continue
        files = sorted([f for f in d.iterdir() if f.is_file()])
        shown = d.relative_to(root) if d != root else Path(".")
        print(f"{'  ' * depth}[dir] {shown}/   ({len(files)} files)")
        for f in files[:files_per_dir]:
            print(f"{'  ' * (depth + 1)}{f.name}")
        if len(files) > files_per_dir:
            print(f"{'  ' * (depth + 1)}... +{len(files) - files_per_dir} more")


def base_id(p: Path) -> str:
    """'100694_sat.jpg' and '100694_mask.png' both map to '100694'."""
    stem = p.stem
    for suffix in ("_mask", "_sat", "_labels", "_label", "_rgb"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Pair every mask image with its satellite image by their shared base id."""
    masks = sorted(p for p in root.rglob("*.png") if "mask" in p.name.lower())
    images = sorted(list(root.rglob("*.jpg")) + list(root.rglob("*.jpeg")))

    image_by_id: dict[str, Path] = {}
    for p in images:
        image_by_id.setdefault(base_id(p), p)     # first match wins

    pairs = [(image_by_id[base_id(m)], m) for m in masks if base_id(m) in image_by_id]
    print(f"\nfound: {len(images)} images, {len(masks)} masks, {len(pairs)} matched pairs")
    if not pairs:
        raise SystemExit("No image/mask pairs found - paste the tree above and stop here.")
    print("example pair:", pairs[0][0].name, "<->", pairs[0][1].name)
    return pairs


# --------------------------------------------------------------------------- #
# 2. Class dictionary -> agriculture colour
# --------------------------------------------------------------------------- #
def load_class_dict(root: Path) -> list[dict]:
    """Print class_dict.csv row by row - this is the ground truth for the colours."""
    csvs = sorted(root.rglob("class_dict*.csv"))
    if not csvs:
        print("\n!! class_dict.csv NOT found in the dataset")
        return []
    path = csvs[0]
    print(f"\n=== 2. class dictionary: {path} ===")

    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    first = [c.strip().lower() for c in raw[0].split(",")]
    has_header = first[0] in {"name", "class_name", "class", "label", "id", "category"}

    rows = []
    if has_header:
        for row in csv.DictReader(raw):
            rows.append({k.strip(): v.strip() for k, v in row.items()})
        for row in rows:
            print(f"  {row}")
    else:                                   # no header: assume name,r,g,b
        for line in raw:
            parts = [c.strip() for c in line.split(",")]
            if len(parts) < 4:
                continue
            rows.append({"name": parts[0], "r": parts[1], "g": parts[2], "b": parts[3]})
            print(f"  {rows[-1]}")
    return rows


def agriculture_rgb(rows: list[dict]) -> tuple[int, int, int]:
    """Find the agriculture row; fall back to DeepGlobe's documented (255,255,0)."""
    for row in rows:
        name = row.get("name", "").lower()
        if "agri" in name:
            rgb = (int(row["r"]), int(row["g"]), int(row["b"]))
            print(f"\nagriculture class '{row['name']}' -> RGB {rgb}")
            return rgb
    print("\n!! no 'agri' row found; using DeepGlobe default RGB (255,255,0)")
    return (255, 255, 0)


def to_binary(mask_rgb: np.ndarray, agri: tuple[int, int, int]) -> np.ndarray:
    """HxWx3 colour mask -> HxW uint8 mask with 1 = farmland, 0 = everything else."""
    # np.all(..., axis=-1) = True where R,G,B all match the agriculture colour
    return np.all(mask_rgb == np.array(agri, dtype=np.uint8), axis=-1).astype(np.uint8)


def print_mask_colours(mask_path: Path) -> None:
    """Show which colours actually occur in a mask, to confirm the mapping is right."""
    rgb = np.asarray(Image.open(mask_path).convert("RGB")).reshape(-1, 3)
    colours, counts = np.unique(rgb, axis=0, return_counts=True)
    order = np.argsort(-counts)[:8]
    print(f"\ntop colours in {mask_path.name} (RGB -> % of pixels):")
    for i in order:
        c = tuple(int(v) for v in colours[i])
        print(f"  {c} -> {100.0 * counts[i] / rgb.shape[0]:5.1f}%")
    print(f"  distinct colours in this mask: {len(colours)}")


# --------------------------------------------------------------------------- #
# 3. Split by image
# --------------------------------------------------------------------------- #
def split_pairs(pairs: list[tuple[Path, Path]], seed: int = SEED):
    """Shuffle IMAGES (not tiles) with a fixed seed, then cut 70/15/15."""
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)          # seeded shuffle -> reproducible split
    n = len(idx)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    splits = {
        "train": [pairs[i] for i in idx[:n_train]],
        "val": [pairs[i] for i in idx[n_train:n_train + n_val]],
        "test": [pairs[i] for i in idx[n_train + n_val:]],   # remainder
    }
    print(f"\n=== 3. split by image (seed={seed}) ===")
    for name, items in splits.items():
        print(f"  {name:5s}: {len(items):4d} images")
    return splits


# --------------------------------------------------------------------------- #
# 4. Tiling
# --------------------------------------------------------------------------- #
def build_tiles(pairs, agri, tile: int = TILE, limit: int | None = None) -> list[dict]:
    """Cut every image into non-overlapping tile x tile tiles. Partial edges are dropped."""
    use = pairs[:limit] if limit else pairs
    tiles: list[dict] = []
    for img_path, mask_path in use:
        img = np.asarray(Image.open(img_path).convert("RGB"))      # HxWx3 uint8
        msk = np.asarray(Image.open(mask_path).convert("RGB"))     # colour mask
        if msk.shape[:2] != img.shape[:2]:                          # rare size mismatch
            msk = np.asarray(
                Image.fromarray(msk).resize((img.shape[1], img.shape[0]), Image.NEAREST)
            )
        mask = to_binary(msk, agri)
        h, w = img.shape[:2]
        for r in range(0, h - tile + 1, tile):        # stride == tile -> no overlap
            for c in range(0, w - tile + 1, tile):
                tiles.append({
                    "image": img[r:r + tile, c:c + tile].copy(),   # copy frees the big image
                    "mask": mask[r:r + tile, c:c + tile].copy(),
                    "src": img_path.name,
                })
    return tiles


# --------------------------------------------------------------------------- #
# 5. PyTorch Dataset (used by Step 4's training loop)
# --------------------------------------------------------------------------- #
class DeepGlobeTiles(Dataset):
    """Wraps the tile list so a DataLoader can pull batches from it."""

    def __init__(self, tiles: list[dict]):
        self.tiles = tiles                 # list of {"image", "mask", "src"}

    def __len__(self) -> int:
        return len(self.tiles)             # DataLoader uses this to know the epoch size

    def __getitem__(self, i: int):
        t = self.tiles[i]
        image = torch.from_numpy(t["image"]).permute(2, 0, 1).float() / 255.0
        # .permute(2,0,1): HxWxC (numpy) -> CxHxW, the layout PyTorch conv layers expect
        # .float() / 255.0: uint8 0..255 -> float 0..1
        mask = torch.from_numpy(t["mask"]).long()   # int64 class ids, needed by CrossEntropyLoss
        return image, mask


# --------------------------------------------------------------------------- #
# 6. Samples + overall farmland share
# --------------------------------------------------------------------------- #
def save_samples(tiles: list[dict], out_path: Path, n: int = 4) -> None:
    """Plot n image/mask pairs side by side."""
    # prefer mixed tiles (some farmland AND some other) over all-zero / all-one ones
    mixed = [t for t in tiles if 0.05 < t["mask"].mean() < 0.95] or tiles
    picks = mixed[:n]
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for j, t in enumerate(picks):
        axes[0, j].imshow(t["image"])
        axes[0, j].set_title(f"image\n{t['src'][:14]}", fontsize=8)
        axes[1, j].imshow(t["mask"], cmap="gray", vmin=0, vmax=1)
        axes[1, j].set_title(f"mask: farm={100 * t['mask'].mean():.1f}%", fontsize=8)
        for ax in (axes[0, j], axes[1, j]):
            ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\nsaved samples -> {out_path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    show_tree(INPUT_ROOT)

    pairs = find_pairs(INPUT_ROOT)
    rows = load_class_dict(INPUT_ROOT)
    agri = agriculture_rgb(rows)
    print_mask_colours(pairs[0][1])

    splits = split_pairs(pairs)

    print("\n=== 4. tiling (small subset first) ===")
    all_tiles = {}
    total_px = 0
    total_farm = 0
    for name, items in splits.items():
        limit = MAX_IMAGES[name]
        tiles = build_tiles(items, agri, limit=limit)
        all_tiles[name] = tiles
        if not tiles:                       # tiny split (e.g. 4 images) can be empty
            print(f"  {name:5s}: 0 tiles - skipped")
            continue
        farm = sum(int(t["mask"].sum()) for t in tiles)
        px = sum(t["mask"].size for t in tiles)
        total_px += px
        total_farm += farm
        print(f"  {name:5s}: {len(items[:limit]):3d} images -> {len(tiles):5d} tiles  "
              f"({int(np.sqrt(len(tiles)/max(len(items[:limit]),1)))}x"
              f"{int(np.sqrt(len(tiles)/max(len(items[:limit]),1)))})  "
              f"farmland {100.0 * farm / px:.1f}%")

    # prove the Dataset wrapper works (this is what Step 4 will feed the U-Net)
    if all_tiles.get("train"):
        ds = DeepGlobeTiles(all_tiles["train"])
        x, y = ds[0]
        print(f"\nDeepGlobeTiles: {len(ds)} tiles | image {tuple(x.shape)} {x.dtype} | "
              f"mask {tuple(y.shape)} {y.dtype}")

    save_samples(all_tiles["train"], OUT_DIR / "step3_samples.png")

    print(f"\n=== 5. OVERALL farmland share: {100.0 * total_farm / max(total_px, 1):.1f}% "
          f"of {total_px:,} labelled pixels ===")


if __name__ == "__main__":
    main()
