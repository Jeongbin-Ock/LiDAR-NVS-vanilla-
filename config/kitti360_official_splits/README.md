# Official KITTI-360 3D semantic split files

These two files were read unchanged on 2026-10-08 from the official
`data_3d_semantics.zip` linked by the
[KITTI-360 download page](https://www.cvlibs.net/datasets/kitti-360/download.php).
`provenance.json` records the archive URL, member paths and SHA-256 digests.
Only 103,736 bytes of ZIP metadata and split contents were fetched with HTTP
byte ranges; the accumulated point clouds were not downloaded.

The files list 239 train / 61 val **accumulated-cloud windows**, whose names
encode inclusive original frame-ID ranges. Adjacent windows overlap. For raw
NVS, use the union of val ranges, match actual `.bin` IDs, remove LiDAR4D's ten
benchmark clips from both splits, and give validation priority over overlapping
train ranges. Raw scans outside the semantic annotation lists remain usable
for bbox-free NVS training.
