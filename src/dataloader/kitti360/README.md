# KITTI-360 data utilities

This package contains NumPy/standard-library data preparation and temporal
selection. It does not import the dataset registry, PyTorch, Utonia or CUDA.

| Module | Responsibility | Consumers |
|---|---|---|
| `metadata.py` | Benchmark clip definitions, timestamp reading, raw sweep inventory | Train/test sequence readers and split preparation |
| `sampling.py` | Physical-time start grid, eligible endpoints, random endpoint and middle GT selection | `KITTI360DynamicDataset` during indexing and access |
| `splits.py` | Official train/val interval parsing, exclusions, contiguous segments and provenance manifest assembly | Split preparation and interval tests |
| `__main__.py` | Config parsing and writing the assembled JSON | Preparation CLI |
| `compact_val.py` | Compact-val window keys, sampling contract, group labels | `KITTI360DynamicDataset` (val), `ModelWrapper` group logging, `tools/build_kitti360_compact_val.py` |

Canonical preparation command, from the repository root:

```bash
python -m src.dataloader.kitti360 --config config/kitti360_train_split.yaml
```

Runtime code imports only `metadata` and `sampling`, so it does not load split
construction. `src/dataloader/__init__.py` loads the dataset registry lazily,
so these torch-free modules stay importable without PyTorch.

Calibration, poses, raw point loading and GT cameras live in `reader.py`
(shared by training and inference) with the HDL-64E laser model in
`hdl64e.py`; the LiDAR4D benchmark windows are `lidar4d_test.py` and the
training dataset is `dynamic.py`. Batch packing is the shared
`src/dataloader/collate.py`. The compact-val manifest
`config/kitti360_compact_val_windows.json` is built by
`tools/build_kitti360_compact_val.py`; see `config/kitti360_train.md`.
