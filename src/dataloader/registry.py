"""``data.dataset_name`` -> dataset class, for training, validation and test."""
from .kitti360.dynamic import KITTI360DynamicDataset
from .nuscenes.dynamic import NuScenesDynamicLiDAR4DDataset
from .nuscenes.legacy import NuScenesNVSDataset
from .waymo.dynamic import WaymoDynamicDataset

dataset_dict = {
    "nuscenes": NuScenesNVSDataset,
    "nuscenes_dynamic_lidar4d": NuScenesDynamicLiDAR4DDataset,
    "kitti360_dynamic": KITTI360DynamicDataset,
    "waymo_dynamic": WaymoDynamicDataset,
}
