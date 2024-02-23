from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

if TYPE_CHECKING:
    from pathlib import Path


import cv2
import numpy as np

from da_od.config import output_img


class MiDaSEstimator:
    """MiDaSEstimator class for depth estimation using MiDaS models.

    This class is initialized with an image path and a model type, then processes the image to estimate depth,
    save both raw and colored depth images, and returns these images as numpy arrays.

    This class encapsulates the functionality of the MiDaS model, allowing for easy
    loading and inference of depth maps from images. It supports different types of
    MiDaS models: 'DPT_Large', 'DPT_Hybrid', and 'MiDaS_small'. 'DPT_Large' offers high accuracy,
    'DPT_Hybrid' provides a balance between performance and resource utilization,
    and 'MiDaS_small' is efficient for limited-resource environments.

    Attributes:
        image_path (Path): Path to the input image.
        device (torch.device): Device for model computation (CPU or CUDA).
        model (torch.nn.Module): Loaded MiDaS model for depth estimation.
        transform (Compose): Transformations for input image processing.
    """

    def __init__(self: MiDaSEstimator, image_path: Path, model_type: str = "DPT_Large") -> None:
        """Initializes MiDaSEstimator with an image path and a specified model type.

        Parameters:
            image_path (Path): Path to the input image.
            model_type (str): Type of MiDaS model ('DPT_Large', 'DPT_Hybrid', 'MiDaS_small').
        """
        self.image_path = image_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device)
        self.model.eval()
        self.transform = Compose(
            [Resize(384), ToTensor(), lambda x: x.to(self.device), lambda x: x.unsqueeze(0)],
        )

    def process_image(self: MiDaSEstimator) -> tuple[np.ndarray, np.ndarray]:
        """Processes the image to estimate depth, saves, and returns depth images.

        Returns:
            tuple[np.ndarray, np.ndarray]: Colored and raw depth images as numpy arrays.
        """
        img = Image.open(self.image_path)
        input_tensor = self.transform(img)

        with torch.no_grad():
            depth = self.model(input_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_numpy = depth.cpu().numpy()

        # Normalize depth for visualization and saving
        depth_min = depth_numpy.min()
        depth_max = depth_numpy.max()
        depth_normalized = (depth_numpy - depth_min) / (depth_max - depth_min) * 255.0
        depth_uint8 = depth_normalized.astype(np.uint8)

        # Save normalized depth image (for visualization)
        raw_depth_image_path = output_img / f"{self.image_path.stem}_depth_raw.jpg"
        cv2.imwrite(str(raw_depth_image_path), depth_uint8)

        # Save raw depth data
        raw_depth_data_path = output_img / f"{self.image_path.stem}_raw_depth.npy"
        np.save(raw_depth_data_path, depth_numpy)

        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        colored_depth_image_path = output_img / f"{self.image_path.stem}_depth_colormap.jpg"
        cv2.imwrite(str(colored_depth_image_path), depth_colored)

        return depth_colored, depth_numpy
