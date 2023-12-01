from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
import logging

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class BoundingBoxUtility:
    """Provides utility functions for handling bounding boxes in image datasets.

    This class includes static methods for calculating bounding boxes from polygons
    and parsing annotation files to extract bounding box data.
    """

    @staticmethod
    def get_bounding_box(polygon: list[tuple[float, float]]) -> list[float]:
        """Calculates the bounding box for a given polygon.

        Args:
            polygon (list[tuple[float, float]]): A list of (x, y) tuples representing the polygon's vertices.

        Returns:
            list[float]: A list containing the coordinates [x_min, y_min, x_max, y_max] of the bounding box.
        """
        x_coords, y_coords = zip(*polygon)
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    @staticmethod
    def parse_annotations(json_file: Path) -> list[list[float]]:
        """Parses a JSON file containing annotations to extract bounding boxes.

        Args:
            json_file (Path): Path to the JSON file with annotations.

        Returns:
            list[list[float]]: A list of bounding boxes, each represented as [x_min, y_min, x_max, y_max].
        """
        with json_file.open() as f:
            data = json.load(f)
        return [
            BoundingBoxUtility.get_bounding_box(obj["polygon"]) for obj in data["objects"] if "polygon" in obj
        ]


class CityscapesDataset(Dataset):
    """A dataset class for handling Cityscapes image data.

    This class extends PyTorch's Dataset and is tailored for the Cityscapes dataset,
    including handling of images and their corresponding annotations.

    Attributes:
        image_dir (Path): Directory containing the image files.
        annotation_dir (Path): Directory containing the annotation files.
        transforms (Optional[torchvision.transforms.Compose]): Transformations to apply to the images.
    """

    def __init__(
        self: CityscapesDataset,
        image_dir: Path,
        annotation_dir: Path,
        transforms: torchvision.transforms.Compose | None = None,
    ) -> None:
        """Initializes the CityscapesDataset object.

        Args:
            image_dir (Path): Directory containing the image files.
            annotation_dir (Path): Directory containing the annotation files.
            transforms (Optional[torchvision.transforms.Compose]): Transformations to apply to the images.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.images = os.listdir(image_dir)

    def __len__(self: CityscapesDataset) -> int:
        """Returns the number of images in the dataset.

        This method is required by the PyTorch Dataset interface.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self: CityscapesDataset, idx: int) -> tuple[Image.Image, dict[str, torch.Tensor]]:
        """Retrieves an image and its corresponding annotations by index.

        This method is required by the PyTorch Dataset interface. It loads the image,
        reads the corresponding annotation, and applies any specified transformations.

        Args:
            idx (int): The index of the image in the dataset.

        Returns:
            tuple[Image.Image, dict[str, torch.Tensor]]: A tuple containing the image and its annotations.
                The annotations include bounding boxes and labels.
        """
        img_path = self.image_dir / self.images[idx]
        img = Image.open(img_path).convert("RGB")
        annotation_file = self.annotation_dir / self.images[idx].replace(
            "_leftImg8bit.png",
            "_gtFine_polygons.json",
        )

        boxes_list = BoundingBoxUtility.parse_annotations(annotation_file)
        boxes_tensor = torch.as_tensor(boxes_list, dtype=torch.float32)
        labels = torch.ones((len(boxes_tensor),), dtype=torch.int64)
        target = {"boxes": boxes_tensor, "labels": labels}
        if self.transforms:
            img = self.transforms(img)
        return img, target


class DataLoaderUtilities:
    """Provides utility functions for data loading in PyTorch.

    This class includes static methods to assist with collating batches of data,
    specifically tailored for datasets with bounding box annotations.
    """

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Custom collate function for batching images along with their bounding box annotations.

        Args:
            batch (list[tuple[torch.Tensor, dict[str, torch.Tensor]]]): A list of tuples,
                where each tuple contains an image tensor and its corresponding annotation data.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: A batch of images and their annotations.
        """
        images, boxes, labels = zip(*[(item[0], item[1]["boxes"], item[1]["labels"]) for item in batch])
        stacked_images = torch.stack(images, dim=0)
        # Convert tuples to lists before padding
        padded_boxes = pad_sequence(list(boxes), batch_first=True, padding_value=-1)
        padded_labels = pad_sequence(list(labels), batch_first=True, padding_value=-1)
        return stacked_images, {"boxes": padded_boxes, "labels": padded_labels}


class ImageDisplayUtility:
    """Provides utility functions for displaying images with bounding boxes.

    This class includes static methods to assist with visualizing images and their
    bounding box annotations using Matplotlib.
    """

    @staticmethod
    def show_image_with_boxes(
        img: Image.Image,
        boxes: list[list[float]],
        ax: matplotlib.axes.Axes | None = None,
    ) -> None:
        """Displays an image with bounding boxes overlaid.

        Args:
            img (Image.Image): The image to display.
            boxes (list[list[float]]): A list of bounding boxes, each represented as
                                       [x_min, y_min, x_max, y_max].
            ax (Optional[matplotlib.axes.Axes]): Matplotlib Axes object to draw the image on.
                If None, a new figure and axes are created.
        """
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
