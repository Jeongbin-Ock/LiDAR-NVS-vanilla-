# KITTI-360 data utilities

This package contains NumPy/standard-library data preparation and temporal
selection. It does not import the dataset registry, PyTorch, Utonia or CUDA.

| Module | Responsibility | Consumers |
|---|---|---|
| `metadata.py` | Benchmark clip definitions, timestamp reading, raw sweep inventory | Train/test sequence readers and split preparation |
| `sampling.py` | Physical-time start grid, eligible endpoints, random endpoint and middle GT selection | `KITTI360DynamicDataset` during indexing and access |
| `splits.py` | Official train/val interval parsing, exclusions, contiguous segments and provenance manifest assembly | Split preparation and interval tests |
| `__main__.py` | Config parsing and writing the assembled JSON | Preparation CLI |

Canonical preparation command, from the repository root:

```bash
python -m src.kitti360 --config config/kitti360_train_split.yaml
```

`src/kitti360_split.py` is a compatibility facade. It re-exports the same public
function objects and retains `python -m src.kitti360_split`. Existing analysis
scripts can continue importing it; new runtime code imports only `metadata`
and `sampling` so it does not load split construction.

GT projection, camera construction and raw point loading stay in
`src/dataloader/kitti360_lidar4d_test.py`; training samples and batch preparation
stay in `src/dataloader/kitti360_dynamic.py`. Compact-val analysis currently lives
in `tmp/kitti_review/`; its CSV and dry-run selection are not runtime manifests.
