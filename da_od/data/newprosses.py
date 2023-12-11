from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

import cv2


class KittiDatasetProcessor:
    """A class to process KITTI dataset images and labels.

    Attributes:
        image_dir (Path): The directory containing the images.
        label_dir (Path): The directory containing the labels.
    """

    def __init__(self: KittiDatasetProcessor, image_dir: str, label_dir: str) -> None:
        """Initialize the KittiDatasetProcessor with the given directories.

        Args:
            image_dir (str): The path to the image directory.
            label_dir (str): The path to the label directory.
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

    def load_labels(self: KittiDatasetProcessor, label_file: str) -> list[dict[str, object]]:
        """Load labels from a label file.

        Args:
            label_file (str): The name of the label file to load.

        Returns:
            list[dict[str, object]]: A list of label data dictionaries.
        """
        labels = []
        label_path = self.label_dir / label_file
        with label_path.open() as file:
            lines = file.readlines()
            for line in lines:
                label_data = self.parse_label(line)
                if label_data:
                    labels.append(label_data)
        return labels

    def parse_label(self: KittiDatasetProcessor, label_line: str) -> dict[str, object] | None:
        """Parse a single line of label data.

        Args:
            label_line (str): The label line to parse.

        Returns:
            Optional[Dict[str, object]]: The parsed label data or None if parsing fails.
        """
        parts = label_line.strip().split(" ")
        if len(parts) >= 15:
            return {
                "type": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": [float(part) for part in parts[4:8]],
                "dimensions": [float(part) for part in parts[8:11]],
                "location": [float(part) for part in parts[11:14]],
                "rotation_y": float(parts[14]),
            }
        return None

    def load_image(self: KittiDatasetProcessor, image_file: str) -> np.ndarray | None:
        """Load an image from a file.

        Args:
            image_file (str): The name of the image file to load.

        Returns:
            np.ndarray | None: The loaded image as a NumPy array, or None if loading fails.
        """
        image_path = self.image_dir / image_file
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        return image

    def draw_bounding_boxes(
        self: KittiDatasetProcessor,
        image: np.ndarray,
        labels: list[dict[str, object]],
    ) -> np.ndarray:
        """Draw bounding boxes on an image based on the provided labels.

        Args:
            image (np.ndarray): The image to draw bounding boxes on.
            labels (List[Dict[str, object]]): A list of label data containing bounding box information.

        Returns:
            np.ndarray: The image with bounding boxes drawn.
        """
        for label in labels:
            bbox = label["bbox"]
            if isinstance(bbox, list) and len(bbox) == 4:
                top_left = (int(bbox[0]), int(bbox[1]))
                bottom_right = (int(bbox[2]), int(bbox[3]))
                image = cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 1)
        return image


# Example usage
image_dir = "../../data/data_object_image/training"
label_dir = "../../data/data_object_label/training"

processor = KittiDatasetProcessor(image_dir, label_dir)
image_file = "000004.png"
label_file = "000004.txt"

labels = processor.load_labels(label_file)
image = processor.load_image(image_file)
if image is not None:
    image_with_boxes = processor.draw_bounding_boxes(image, labels)
