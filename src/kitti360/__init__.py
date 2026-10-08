"""Lightweight KITTI-360 metadata, split preparation and temporal sampling.

Submodules are imported explicitly so runtime sampling does not load manifest
preparation, dataset classes, PyTorch, Utonia or CUDA extensions.
"""
