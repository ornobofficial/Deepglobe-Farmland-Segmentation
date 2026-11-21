"""Random augmentations for training tiles - numpy only, no extra dependency.

Why: with 120 images the model overfits from ~epoch 11, so we synthesise variety.
Flips and 90-degree rotations are safe here because satellite images have no
canonical orientation and farmland stays farmland under any rotation.
Applied to the image AND the mask with the SAME random choice, never to val/test.
"""

import numpy as np                                          # array maths
import torch                                                # tensor conversion
from torch.utils.data import Dataset                        # base class we wrap


def random_flip_rot90(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Apply one random combination of h-flip / v-flip / 90-degree rotation to both arrays."""
    if rng.random() < 0.5:                 # horizontal flip
        image = image[:, ::-1]
        mask = mask[:, ::-1]
    if rng.random() < 0.5:                 # vertical flip
        image = image[::-1]
        mask = mask[::-1]
    k = int(rng.integers(0, 4))            # 0..3 quarter turns
    if k:
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class AugDataset(Dataset):
    """Wraps DeepGlobeTiles and augments every sample on the fly (training only)."""

    def __init__(self, base: Dataset, seed: int = 42):
        self.base = base                                     # the plain tile dataset
        self.rng = np.random.default_rng(seed)               # fixed seed -> reproducible

    def __len__(self) -> int:
        return len(self.base)                                # same number of tiles

    def __getitem__(self, i: int):
        tile = self.base.tiles[i]                            # one dict: image / mask / src
        image, mask = random_flip_rot90(tile["image"], tile["mask"], self.rng)
        # same conversion as DeepGlobeTiles.__getitem__, done here so dataset.py stays untouched
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0   # HxWxC -> CxHxW, 0..1
        mask = torch.from_numpy(mask).long()                 # int64 class ids for CrossEntropy
        return image, mask
