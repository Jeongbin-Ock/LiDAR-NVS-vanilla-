"""Datasets, batch packing and the Lightning DataModule.

Shared code sits at this level (``collate``, ``datamodule``, ``lidar4d_test``,
``registry``); each dataset has its own package (``nuscenes/``, ``kitti360/``).
``dataset_dict`` is loaded on first use, so importing a light submodule such as
``src.dataloader.kitti360.splits`` does not import every dataset.
"""


def __getattr__(name):
    if name == "dataset_dict":
        from .registry import dataset_dict

        return dataset_dict
    raise AttributeError(name)
