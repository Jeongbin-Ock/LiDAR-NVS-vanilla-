"""KITTI-360: sensor reader, calibrated HDL-64E GT, LiDAR4D benchmark windows,
training dataset, and the torch-free split / sampling / compact-val utilities.

Submodules are imported explicitly. ``metadata``, ``sampling``, ``splits`` and
``compact_val`` need only NumPy and the standard library, so split preparation
(``python -m src.dataloader.kitti360``) loads no PyTorch, Utonia or CUDA code.
"""
