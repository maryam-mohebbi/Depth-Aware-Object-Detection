from __future__ import annotations

import json
import os

import matplotlib.axes
import matplotlib.pyplot as plt
import torch
import torchvision
from matplotlib import patches
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Define bounding box and annotation parsing functions
def get_bounding_box(polygon: list[tuple[float, float]]) -> list[float]:
    # Extract coordinates
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    # Calculate bounding box
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return [x_min, y_min, x_max, y_max]


def parse_annotations(json_file: str) -> list[list[float]]:
    with open(json_file) as f:
        data = json.load(f)
    boxes = []
    for obj in data["objects"]:
        if "polygon" in obj:
            box = get_bounding_box(obj["polygon"])
            boxes.append(box)
    return boxes


# Custom Dataset class for Cityscapes
class CityscapesDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        transforms: torchvision.transforms.Compose | None = None,
    ) -> None:
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict[str, torch.Tensor]]:
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        # Load annotations
        annotation_file = os.path.join(
            self.annotation_dir,
            self.images[idx].replace("_leftImg8bit.png", "_gtFine_polygons.json"),
        )

        # Print out paths to verify correctness
        print(f"Image path: {img_path}")
        print(f"Annotation file: {annotation_file}")

        boxes_list = parse_annotations(annotation_file)

        # Convert boxes into a torch.Tensor
        boxes_tensor = torch.as_tensor(boxes_list, dtype=torch.float32)

        # Labels (assuming a single class)
        labels = torch.ones((len(boxes_tensor),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    images = [item[0] for item in batch]
    boxes = [item[1]["boxes"] for item in batch]
    labels = [item[1]["labels"] for item in batch]

    stacked_images = torch.stack(images, dim=0)

    # Pad the bounding boxes and labels to have the same number of objects
    # In this case, we assume that the number of objects is less than a fixed number, e.g., 200
    padded_boxes = pad_sequence(
        boxes,
        batch_first=True,
        padding_value=-1,
    )  # padding value -1 will be ignored in loss computation
    padded_labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=-1,
    )  # padding value -1 will be ignored in loss computation

    # Create a dictionary for the targets
    targets = {}
    targets["boxes"] = padded_boxes
    targets["labels"] = padded_labels

    return stacked_images, targets


# Data transformations
data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ],
)

# Data loader
dataset = CityscapesDataset(
    image_dir="/Users/maryam/projects/depth-aware-object-detection/data/leftImg8bit_trainvaltest/leftImg8bit/train/aachen",
    annotation_dir="/Users/maryam/projects/depth-aware-object-detection/data/gtFine_trainvaltest/gtFine/train/aachen",
    transforms=data_transforms,
)


data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


def show_image_with_boxes(
    img: Image.Image,
    boxes: list[list[float]],
    ax: matplotlib.axes.Axes | None = None,
) -> None:
    """Function to display an image with bounding boxes"""
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


# Get a sample image and its corresponding bounding boxes
sample_img, target = dataset[2]  # Adjust index to view different samples
sample_boxes = target["boxes"].numpy()

# Convert the tensor image back to PIL for display
sample_img_pil = transforms.ToPILImage()(sample_img)

# Display the image with bounding boxes
show_image_with_boxes(sample_img_pil, sample_boxes)

# Print some details about the dataset
print(f"Total images in the dataset: {len(dataset)}")
sample_img_path = dataset.image_dir + dataset.images[0]  # Adjust index to view different samples
print(f"Sample image path: {sample_img_path}")
print(f"Sample bounding boxes: {sample_boxes}")


# Iterate over a few batches
for i, (images, targets) in enumerate(data_loader):
    print(f"Batch {i+1}")
    print(f"Images shape: {images.shape}")
    print(f"Targets: {targets}")
    if i == 1:  # Check 2 batches
        break
