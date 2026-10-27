import lightning.pytorch as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, cfg):
        super().__init__()  
        self.dataset = dataset
        self.cfg = cfg  

    def _collate_fn(self):
        """Every registered dataset declares its own batch packing."""
        return self.dataset.collate_fn

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset(
                cfg=self.cfg.data,
                split=getattr(self.cfg.data, "train_split", "train"),
            ),
            batch_size=self.cfg.train.batch_size,  
            num_workers=self.cfg.data.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self._collate_fn(),
        )

    def _compact_val_subset(self):
        """Path of the compact-val manifest, or None for full-split validation.

        The dataset interprets the manifest: nuScenes and KITTI-360 select
        windows; Waymo selects sequences before applying its window limit.
        """
        compact = getattr(self.cfg.data, "compact_val", None)
        if compact is None or not bool(getattr(compact, "enable", False)):
            return None
        path = getattr(compact, "path", None)
        if not path:
            raise ValueError(
                "data.compact_val.enable=true requires data.compact_val.path"
            )
        return str(path)

    def val_dataloader(self):  
        return DataLoader(
            dataset=self.dataset(
                cfg=self.cfg.data,
                split=getattr(self.cfg.data, "eval_split", "val"),
                window_subset=self._compact_val_subset(),
            ),
            batch_size=self.cfg.test.batch_size,  
            num_workers=int(getattr(self.cfg.data, "eval_num_workers", 0)),
            shuffle=False,
            pin_memory=True,
            collate_fn=self._collate_fn(),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset(
                cfg=self.cfg.data,
                split=getattr(self.cfg.data, "test_split", "test"),
            ),
            batch_size=self.cfg.test.batch_size,  
            num_workers=int(getattr(self.cfg.data, "eval_num_workers", 0)),
            shuffle=False,
            pin_memory=True,
            collate_fn=self._collate_fn(),
        )
