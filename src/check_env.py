"""Step 1 - environment check.

Run this in the Kaggle notebook (GPU T4 x2, Internet on) BEFORE any training:

    !python /kaggle/working/src/check_env.py

It answers two questions: which Python/PyTorch versions are installed,
and is the GPU actually usable (not just "flagged as available")?
"""

import platform
import sys

import torch          # the deep-learning library we train with
import torchvision    # image models/transforms that come with PyTorch


def main() -> None:
    print("=== Step 1: environment check ===")
    print("Python      :", sys.version.split()[0])
    print("Platform    :", platform.platform())

    print("PyTorch     :", torch.__version__)        # torch.__version__ = installed PyTorch version
    print("torchvision :", torchvision.__version__)  # must match the PyTorch version family
    print("CUDA build  :", torch.version.cuda)       # CUDA version PyTorch was compiled against (None = CPU-only build)

    # torch.cuda.is_available() -> True only if a working CUDA driver + GPU is visible
    gpu_on = torch.cuda.is_available()
    print("GPU present :", gpu_on)
    print("Device count:", torch.cuda.device_count())  # how many GPUs the notebook got (T4 x2 -> 2)

    if not gpu_on:
        print(">>> GPU OFF - set Notebook > Settings > Accelerator = GPU T4 x2")
        return

    print("GPU name    :", torch.cuda.get_device_name(0))  # 0 = first GPU
    props = torch.cuda.get_device_properties(0)            # VRAM etc. for GPU 0
    print("GPU memory  :", round(props.total_memory / 1e9, 1), "GB")

    # Allocate a real tensor ON the GPU and do a matmul: proves CUDA computes,
    # not merely that the flag is set. device="cuda" moves the tensor to GPU memory.
    x = torch.rand(1000, 1000, device="cuda")               # random 1000x1000 matrix on GPU
    y = x @ x                                              # matrix multiply on GPU
    torch.cuda.synchronize()                                # wait for the GPU to finish before timing/printing
    print("GPU matmul  :", tuple(y.shape), "| device:", y.device)  # tuple(shape) prints (1000, 1000)

    # Free the test tensors so they do not sit in VRAM during training
    del x, y
    torch.cuda.empty_cache()                                # release cached GPU memory back to the driver
    print("=== OK: GPU is usable ===")


if __name__ == "__main__":
    main()
