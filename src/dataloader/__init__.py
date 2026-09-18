from .nuscene import NuScenesNVSDataset
from .nuscene_dynamic_lidar4d import NuScenesDynamicLiDAR4DDataset
#from once import ONCEDataset 


dataset_dict = {'nuscenes': NuScenesNVSDataset,
                'nuscenes_dynamic_lidar4d': NuScenesDynamicLiDAR4DDataset,
                #'once': ONCEDataset
                }
