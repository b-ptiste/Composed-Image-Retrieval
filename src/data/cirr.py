import json
from collections import defaultdict
from pathlib import Path

import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.transforms import transform_test, transform_train
from src.data.utils import id2int, pre_caption

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning
import numpy as np

import numpy as np
from torch.utils.data import Sampler

import numpy as np
from torch.utils.data.sampler import Sampler


# own function
class CustomSampler_exclude(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def _create_batches(self):
        member_ids = list(self.data_source.members2pairid.keys())
        member_indices = {
            member_id: self.data_source.members2pairid[member_id]
            for member_id in member_ids
        }

        for indices in member_indices.values():
            np.random.shuffle(indices)

        batches = []
        while any(member_indices.values()):
            np.random.shuffle(member_ids)
            current_batch = []
            for member_id in member_ids:
                if member_indices[member_id]:
                    current_batch.append(member_indices[member_id].pop())
                if len(current_batch) >= self.batch_size:
                    break
            if current_batch:
                batches.append(current_batch)

        np.random.shuffle(batches)
        return batches

    def __iter__(self):
        for batch in self.batches:
            for idx in batch:
                yield idx

    def __len__(self):
        return sum(len(batch) for batch in self.batches)


# own function
class CustomSampler_include(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def _create_batches(self):
        member_ids = list(self.data_source.members2pairid.keys())
        np.random.shuffle(member_ids)

        batches = []
        current_batch = []
        for member_id in member_ids:
            group_indices = self.data_source.members2pairid[member_id]

            np.random.shuffle(group_indices)

            current_batch.extend(group_indices)
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []

        np.random.shuffle(batches)
        return batches

    def __iter__(self):
        a = []
        for batch in self.batches:
            for idx in batch:
                yield idx

    def __len__(self):
        return sum(len(batch) for batch in self.batches)


import numpy as np
from torch.utils.data import Sampler


class CustomMixedSampler(Sampler):
    def __init__(self, data_source, batch_size, alpha):
        self.data_source = data_source
        self.batch_size = batch_size
        self.alpha = alpha
        self.batches = self._create_batches()

    def _create_batches(self):
        member_ids = list(self.data_source.members2pairid.keys())
        member_indices = {
            member_id: self.data_source.members2pairid[member_id][:]
            for member_id in member_ids
        }

        batches = []
        while any(member_indices.values()):
            if np.random.binomial(1, self.alpha):
                # Exclude approach
                np.random.shuffle(member_ids)
                current_batch = []
                for member_id in member_ids:
                    if member_indices[member_id]:
                        current_batch.append(member_indices[member_id].pop(0))
                    if len(current_batch) >= self.batch_size:
                        break
                if current_batch:
                    batches.append(current_batch)

            else:
                # Include approach
                np.random.shuffle(member_ids)
                current_batch = []
                members_processed = []
                for member_id in member_ids:
                    if member_indices[member_id]:
                        group_indices = member_indices[member_id]
                        np.random.shuffle(group_indices)
                        current_batch.extend(group_indices)
                        members_processed.append(member_id)
                        if len(current_batch) >= self.batch_size:
                            break
                for member_id in members_processed:
                    member_indices[member_id] = []
                if current_batch:
                    batches.append(current_batch)

        np.random.shuffle(batches)
        return batches

    def __iter__(self):
        a = []
        for batch in self.batches:
            for idx in batch:
                yield idx

    def __len__(self):
        return sum(len(batch) for batch in self.batches)


class CIRRDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        annotation: dict = {"train": "", "val": ""},
        img_dirs: dict = {"train": "", "val": ""},
        emb_dirs: dict = {"train": "", "val": ""},
        image_size: int = 384,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform_train = transform_train(image_size)
        self.transform_test = transform_test(image_size)

        self.data_train = CIRRDataset(
            transform=self.transform_train,
            annotation=annotation["train"],
            img_dir=img_dirs["train"],
            emb_dir=emb_dirs["train"],
            split="train",
        )

        # own code
        self._type = "mix"

        if self._type == "exclude":
            self.sampler_data_train = CustomSampler_exclude(
                self.data_train, self.batch_size
            )
        elif self._type == "include":
            self.sampler_data_train = CustomSampler_include(
                self.data_train, self.batch_size
            )
        elif self._type == "mix":
            self.sampler_data_train = CustomMixedSampler(
                self.data_train,
                self.batch_size,
                alpha=0.6,  # TODO hard coded to be added to yaml
            )
        self.data_val = CIRRDataset(
            transform=self.transform_test,
            annotation=annotation["val"],
            img_dir=img_dirs["val"],
            emb_dir=emb_dirs["val"],
            split="val",
        )

    def prepare_data(self):
        # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
        # download data, pre-process, split, save to disk, etc...
        pass

    def train_dataloader(self):
        if self._type in ["exclude", "include", "mix"]:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                # shuffle=True,
                drop_last=True,
                sampler=self.sampler_data_train,
            )
        else:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
                drop_last=True,
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class CIRRTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        annotation: str,
        img_dirs: str,
        emb_dirs: str,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 384,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform_test = transform_test(image_size)

        self.data_test = CIRRDataset(
            transform=self.transform_test,
            annotation=annotation,
            img_dir=img_dirs,
            emb_dir=emb_dirs,
            split="test",
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class CIRRDataset(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
        img_dir: str,
        emb_dir: str,
        split: str,
        max_words: int = 30,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.annotation_pth = annotation
        assert Path(annotation).exists(), f"Annotation file {annotation} does not exist"
        self.annotation = json.load(open(annotation, "r"))
        self.split = split
        self.max_words = max_words
        self.img_dir = Path(img_dir)
        self.emb_dir = Path(emb_dir)
        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid split: {split}, must be one of train, val, or test"
        assert self.img_dir.exists(), f"Image directory {img_dir} does not exist"
        assert self.emb_dir.exists(), f"Embedding directory {emb_dir} does not exist"

        self.pairid2ref = {
            ann["pairid"]: id2int(ann["reference"]) for ann in self.annotation
        }
        self.int2id = {
            id2int(ann["reference"]): ann["reference"] for ann in self.annotation
        }

        ids = {ann["reference"] for ann in self.annotation}
        assert len(self.int2id) == len(ids), "Reference ids are not unique"

        self.pairid2members = {
            ann["pairid"]: id2int(ann["img_set"]["members"]) for ann in self.annotation
        }

        # own code
        self.pairid2int = {ann["pairid"]: i for i, ann in enumerate(self.annotation)}
        self.members2pairid = defaultdict(list)

        for pairid, members in self.pairid2members.items():
            for member in members:
                self.members2pairid[member].append(self.pairid2int[pairid])
                break

        if split != "test":
            self.pairid2tar = {
                ann["pairid"]: id2int(ann["target_hard"]) for ann in self.annotation
            }
        else:
            self.pairid2tar = None

        if split == "train":
            img_pths = self.img_dir.glob("*/*.png")
            emb_pths = self.emb_dir.glob("*/*.pth")
        else:
            img_pths = self.img_dir.glob("*.png")
            emb_pths = self.emb_dir.glob("*.pth")
        self.id2imgpth = {img_pth.stem: img_pth for img_pth in img_pths}
        self.id2embpth = {emb_pth.stem: emb_pth for emb_pth in emb_pths}

        for ann in self.annotation:
            assert (
                ann["reference"] in self.id2imgpth
            ), f"Path to reference {ann['reference']} not found in {self.img_dir}"
            assert (
                ann["reference"] in self.id2embpth
            ), f"Path to reference {ann['reference']} not found in {self.emb_dir}"
            if split != "test":
                assert (
                    ann["target_hard"] in self.id2imgpth
                ), f"Path to target {ann['target_hard']} not found"
                assert (
                    ann["target_hard"] in self.id2embpth
                ), f"Path to target {ann['target_hard']} not found"

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        reference_img_pth = self.id2imgpth[ann["reference"]]
        reference_img = Image.open(reference_img_pth).convert("RGB")
        reference_img = self.transform(reference_img)

        caption = pre_caption(ann["caption"], self.max_words)

        if self.split == "test":
            reference_feat = torch.load(self.id2embpth[ann["reference"]])
            return reference_img, reference_feat, caption, ann["pairid"]

        target_emb_pth = self.id2embpth[ann["target_hard"]]
        target_feat = torch.load(target_emb_pth).cpu()

        return (
            reference_img,
            target_feat,
            caption,
            ann["pairid"],
        )
