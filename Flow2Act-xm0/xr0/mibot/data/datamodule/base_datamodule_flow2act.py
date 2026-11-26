# Copyright (C) 2026 Xiaomi Corporation.
from copy import deepcopy

from lightning import LightningDataModule
from mmengine import Config, DATASETS
import torch
from torch.utils.data import DataLoader

from mibot.data.collate.custom_collate_flow2act import CustomCollateFlow2Act


@DATASETS.register_module()
class BaseDataModuleFlow2Act(LightningDataModule):
    """Four-suite shared LIBERO LeRobot loader for the stage experiments."""

    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params: Config = params
        # Isolate shuffle/worker seeding from stage-specific model initialization.
        self._data_generator = torch.Generator().manual_seed(int(params.get("seed", 42)))
        self.batch_size: int = params.train_datasets.get("batch_size", 16)
        self.collate_fn = CustomCollateFlow2Act(processor_path=params.get("processor_path"))

    def train_dataloader(self) -> DataLoader:
        dataset_type = self.params.get("type", "libero_lerobot")
        if dataset_type == "libero_lerobot":
            from mibot.data.datasets.libero_dataset_flow2act import LiberoDatasetFlow2Act

            train_set = LiberoDatasetFlow2Act(deepcopy(self.params))
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        num_workers = int(self.params.get("num_workers", 16))
        worker_options = {}
        if num_workers > 0:
            worker_options = {
                "prefetch_factor": int(self.params.get("prefetch_factor", 8)),
                "persistent_workers": True,
            }
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self._data_generator,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            **worker_options,
        )

    def val_dataloader(self) -> list:
        return []
