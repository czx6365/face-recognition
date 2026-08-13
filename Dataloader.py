"""CelebA multi-label facial-attribute dataset utilities.

The project predicts the 40 binary attributes provided by CelebA.  The data
root can be supplied explicitly or through the CELEBA_ROOT environment
variable, so the repository no longer depends on a machine-specific path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SIZE = (156, 128)

DefaultTransform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

TrainTransform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


class CelebADataset(Dataset):
    """CelebA dataset for 40-attribute multi-label prediction.

    Expected layout (``data_root`` may point either to the outer CelebA
    directory or directly to ``Dataset``)::

        CelebA/
          Dataset/
            Eval/list_eval_partition.txt
            Anno/list_attr_celeba.txt
            Img/img_align_celeba/*.jpg

    CelebA uses -1/1 attribute labels. They are converted to 0/1 tensors for
    binary multi-label learning.
    """

    SPLIT_TO_PARTITION = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        transform=DefaultTransform,
        train_mode: str = "train",
        data_root: Optional[str] = None,
    ) -> None:
        if train_mode not in self.SPLIT_TO_PARTITION:
            raise ValueError(
                f"Invalid train_mode={train_mode!r}; choose from "
                f"{sorted(self.SPLIT_TO_PARTITION)}"
            )

        self.transform = transform
        self.dataset_dir = self._resolve_dataset_dir(data_root)
        self.file_path = {
            "eval_partition": self.dataset_dir / "Eval" / "list_eval_partition.txt",
            "anno_list_attr": self.dataset_dir / "Anno" / "list_attr_celeba.txt",
            "img_dir": self.dataset_dir / "Img" / "img_align_celeba",
        }
        self._validate_paths()

        self.data, self.attribute_names = self._load_data()
        partition_id = self.SPLIT_TO_PARTITION[train_mode]
        self.current_data = (
            self.data[self.data["partition"] == partition_id]
            .reset_index(drop=True)
        )

    @staticmethod
    def _resolve_dataset_dir(data_root: Optional[str]) -> Path:
        root_value = data_root or os.getenv("CELEBA_ROOT") or "CelebA"
        root = Path(root_value).expanduser().resolve()
        dataset_dir = root / "Dataset"
        return dataset_dir if dataset_dir.exists() else root

    def _validate_paths(self) -> None:
        missing = [str(path) for path in self.file_path.values() if not path.exists()]
        if missing:
            formatted = "\n  - ".join(missing)
            raise FileNotFoundError(
                "CelebA files were not found. Set --data-root or CELEBA_ROOT. "
                f"Missing:\n  - {formatted}"
            )

    def _load_data(self):
        partition = pd.read_csv(
            self.file_path["eval_partition"],
            sep=r"\s+",
            header=None,
            names=["img_id", "partition"],
            dtype={"img_id": str, "partition": int},
        )

        attr_path = self.file_path["anno_list_attr"]
        with attr_path.open("r", encoding="utf-8") as handle:
            _ = handle.readline()  # number of images
            attribute_names: List[str] = handle.readline().strip().split()

        attr = pd.read_csv(
            attr_path,
            sep=r"\s+",
            skiprows=2,
            header=None,
            names=["img_id", *attribute_names],
        )
        attr["img_id"] = attr["img_id"].astype(str)

        merged = pd.merge(
            partition,
            attr,
            on="img_id",
            how="inner",
            validate="one_to_one",
        )
        return merged, attribute_names

    def __len__(self) -> int:
        return len(self.current_data)

    def __getitem__(self, idx: int):
        row = self.current_data.iloc[idx]
        image_path = self.file_path["img_dir"] / row["img_id"]

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        labels = row[self.attribute_names].to_numpy(dtype="float32")
        labels = torch.from_numpy((labels > 0).astype("float32"))
        return image, labels


if __name__ == "__main__":
    dataset = CelebADataset(train_mode="train")
    image, labels = dataset[0]
    print(f"samples={len(dataset)}")
    print(f"image_shape={tuple(image.shape)}")
    print(f"num_attributes={labels.numel()}")
