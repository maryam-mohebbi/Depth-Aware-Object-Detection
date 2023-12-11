import glob
import os

import cv2
import numpy as np


class KittiDatasetProcessor:
    def __init__(self, image_dir: str, label_dir: str):
        self.image_dir = image_dir
        self.label_dir = label_dir

    def load_labels(self, label_file: str) -> list[dict]:
        labels = []
        label_path = os.path.join(self.label_dir, label_file)
        with open(label_path) as file:
            lines = file.readlines()
            for line in lines:
                label_data = self.parse_label(line)
                if label_data:
                    labels.append(label_data)
        return labels

    def parse_label(self, label_line: str) -> dict:
        parts = label_line.strip().split(" ")
        if len(parts) > 15:
            label_data = {
                "type": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                "dimensions": [float(parts[8]), float(parts[9]), float(parts[10])],
                "location": [float(parts[11]), float(parts[12]), float(parts[13])],
                "rotation_y": float(parts[14]),
            }
            return label_data
        else:
            return {}

    def load_image(self, image_file: str) -> np.ndarray:
        image_path = os.path.join(self.image_dir, image_file)
        image = cv2.imread(image_path)
        return image

    def attach_labels_to_image(self, image: np.ndarray, labels: list[dict]) -> np.ndarray:
        for label in labels:
            bbox = label["bbox"]
            top_left = (int(bbox[0]), int(bbox[1]))
            bottom_right = (int(bbox[2]), int(bbox[3]))
            image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        return image

    def process_all_images(self) -> None:
        # Find all images in the image directory
        image_files = glob.glob(os.path.join(self.image_dir, "*.png"))

        for image_file in image_files:
            # Extract the file name without the extension to find the corresponding label file
            base_name = os.path.basename(image_file)
            label_file = base_name.replace(".png", ".txt")

            # Load labels and image
            labels = self.load_labels(label_file)
            image = self.load_image(base_name)

            # If image is not found, continue to the next one
            if image is None:
                print(f"Image {base_name} not found.")
                continue

            # Attach labels to image
            image_with_labels = self.attach_labels_to_image(image, labels)

            # Save the image with labels to a file
            output_path = os.path.join(self.image_dir, base_name.replace(".png", "_with_boxes.png"))
            cv2.imwrite(output_path, image_with_labels)
            print(f"Processed and saved {output_path}")


image_dir = "../../data/data_object_image/training/000000"
label_dir = "../../data/data_object_label/training/000000"

# Create an instance of the processor
processor = KittiDatasetProcessor(image_dir, label_dir)

# Process all images and save the output
processor.process_all_images()
