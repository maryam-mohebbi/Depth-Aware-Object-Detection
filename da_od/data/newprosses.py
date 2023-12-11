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

    def load_image(
        self: KittiDatasetProcessor,
        image_file: str,
    ) -> tuple[np.ndarray | None, float, tuple[int, int]]:
        """Load and resize an image from a file, adding padding to maintain aspect ratio.

        Args:
            image_file (str): Name of the image file to load.

        Returns:
            Tuple[np.ndarray | None, float, Tuple[int, int]]:
                - The resized and padded image (or None if loading fails),
                - The scale factor used for resizing,
                - The padding added (left, top).
        """
        target_size = (512, 512)  # Example target size
        image_path = self.image_dir / image_file
        image = cv2.imread(str(image_path))

        if image is None:
            # Returning a tuple with None, 0.0 for scale, and (0, 0) for padding
            return None, 0.0, (0, 0)

        # Calculate aspect ratio
        h, w = image.shape[:2]
        scale = min(target_size[1] / h, target_size[0] / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))

        # Add padding
        top = (target_size[1] - new_h) // 2
        bottom = target_size[1] - new_h - top
        left = (target_size[0] - new_w) // 2
        right = target_size[0] - new_w - left
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT)

        return padded_image, scale, (left, top)

    def draw_bounding_boxes(
        self: KittiDatasetProcessor,
        image: np.ndarray,
        labels: list[dict[str, object]],
        scale: float,
        padding: tuple[int, int],
    ) -> np.ndarray:
        """Draw bounding boxes on a resized image, adjusting for padding.

        Args:
            image (np.ndarray): The resized and padded image.
            labels (List[Dict[str, object]]): List of label data with bounding box info.
            scale (float): The scale factor used in resizing the image.
            padding (Tuple[int, int]): The padding added to the image (left, top).

        Returns:
            np.ndarray: The image with bounding boxes drawn.
        """
        left_pad, top_pad = padding
        for label in labels:
            bbox = label["bbox"]
            if isinstance(bbox, list) and len(bbox) == 4:
                # Scale and adjust the bounding box coordinates
                scaled_bbox = [
                    int(bbox[0] * scale) + left_pad,
                    int(bbox[1] * scale) + top_pad,
                    int(bbox[2] * scale) + left_pad,
                    int(bbox[3] * scale) + top_pad,
                ]
                top_left = (scaled_bbox[0], scaled_bbox[1])
                bottom_right = (scaled_bbox[2], scaled_bbox[3])
                image = cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 1)
        return image


# Example usage
image_dir = "../../data/data_object_image/training"
label_dir = "../../data/data_object_label/training"

processor = KittiDatasetProcessor(image_dir, label_dir)
image_file = "000045.png"
label_file = "000045.txt"

labels = processor.load_labels(label_file)
image, scale, padding = processor.load_image(image_file)
if image is not None:
    image_with_boxes = processor.draw_bounding_boxes(image, labels, scale, padding)

# For saving the image
output_image_file = "output_image_000045.png"
if image_with_boxes is not None:
    cv2.imwrite(output_image_file, image_with_boxes)
    print(f"Output image saved as {output_image_file}")
else:
    print("Failed to process the image or no bounding boxes found.")
