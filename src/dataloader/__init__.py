from .nuscene import NuScenesNVSDataset
from .nuscene_dynamic_lidar4d import NuScenesDynamicLiDAR4DDataset
from .kitti360_dynamic import KITTI360DynamicDataset
#from once import ONCEDataset 


dataset_dict = {'nuscenes': NuScenesNVSDataset,
                'nuscenes_dynamic_lidar4d': NuScenesDynamicLiDAR4DDataset,
                'kitti360_dynamic': KITTI360DynamicDataset,
                #'once': ONCEDataset
                }
