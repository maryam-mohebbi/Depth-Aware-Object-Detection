from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import torch
import torchvision
from matplotlib import patches
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from da_od.config import annotation_path, image_path


class BoundingBoxUtility:
    @staticmethod
    def get_bounding_box(polygon: list[tuple[float, float]]) -> list[float]:
        x_coords, y_coords = zip(*polygon)
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    @staticmethod
    def parse_annotations(json_file: str) -> list[list[float]]:
        with open(json_file) as f:
            data = json.load(f)
        return [
            BoundingBoxUtility.get_bounding_box(obj["polygon"]) for obj in data["objects"] if "polygon" in obj
        ]


class CityscapesDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        annotation_dir: Path,
        transforms: torchvision.transforms.Compose | None = None,
    ):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict[str, torch.Tensor]]:
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        annotation_file = os.path.join(
            self.annotation_dir,
            self.images[idx].replace("_leftImg8bit.png", "_gtFine_polygons.json"),
        )
        boxes_list = BoundingBoxUtility.parse_annotations(annotation_file)
        boxes_tensor = torch.as_tensor(boxes_list, dtype=torch.float32)
        labels = torch.ones((len(boxes_tensor),), dtype=torch.int64)
        target = {"boxes": boxes_tensor, "labels": labels}
        if self.transforms:
            img = self.transforms(img)
        return img, target


class DataLoaderUtilities:
    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        images, boxes, labels = zip(*[(item[0], item[1]["boxes"], item[1]["labels"]) for item in batch])
        stacked_images = torch.stack(images, dim=0)
        # Convert tuples to lists before padding
        padded_boxes = pad_sequence(list(boxes), batch_first=True, padding_value=-1)
        padded_labels = pad_sequence(list(labels), batch_first=True, padding_value=-1)
        return stacked_images, {"boxes": padded_boxes, "labels": padded_labels}


class ImageDisplayUtility:
    @staticmethod
    def show_image_with_boxes(
        img: Image.Image,
        boxes: list[list[float]],
        ax: matplotlib.axes.Axes | None = None,
    ) -> None:
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.imshow(img)
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        plt.show()


# Main execution block
data_transforms = transforms.Compose([transforms.ToTensor()])
dataset = CityscapesDataset(
    image_dir=image_path / "train/aachen",
    annotation_dir=annotation_path / "train/aachen",
    transforms=data_transforms,
)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=DataLoaderUtilities.collate_fn)
sample_img, target = dataset[0]
sample_boxes = target["boxes"].numpy()
sample_img_pil = transforms.ToPILImage()(sample_img)
ImageDisplayUtility.show_image_with_boxes(sample_img_pil, sample_boxes)
