"""StarCraft Motion DataModule."""

from typing import List, Optional

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch_geometric.loader import DataLoader

from src.starcraft.datasets.sc_dataset import SCDataset

from .sc_target_builder import SCTargetBuilderTrain, SCTargetBuilderVal


class SCDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        min_future_alive: int = 8,
        train_map_names: Optional[List[str]] = None,
        val_map_names: Optional[List[str]] = None,
        test_map_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.min_future_alive = min_future_alive
        self.train_map_names = train_map_names
        self.val_map_names = val_map_names
        self.test_map_names = test_map_names

        self.train_transform = SCTargetBuilderTrain(min_future_alive)
        self.val_transform = SCTargetBuilderVal()
        self.test_transform = SCTargetBuilderVal()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = SCDataset(
                self.dataset_root, "train", transform=self.train_transform,
                map_names=self.train_map_names,
            )
            self.val_dataset = SCDataset(
                self.dataset_root, "val", transform=self.val_transform,
                map_names=self.val_map_names,
            )
        elif stage == "validate":
            self.val_dataset = SCDataset(
                self.dataset_root, "val", transform=self.val_transform,
                map_names=self.val_map_names,
            )
        elif stage == "test":
            self.test_dataset = SCDataset(
                self.dataset_root, "test", transform=self.test_transform,
                map_names=self.test_map_names,
            )
        else:
            raise ValueError(f"{stage} should be one of [fit, validate, test]")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
