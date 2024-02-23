from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms import Compose

from da_od.config import output_img
from da_od.depth_anything.dpt import DepthAnything
from da_od.depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize


class DepthAnythingEstimator:
    """DepthAnythingEstimator leverages DepthAnything's pre-trained models for depth estimation.

    This class supports different encoder backbones ("vits", "vitb", and "vitl") to provide flexibility in
    performance and accuracy for depth estimation tasks. It is designed to work specifically with the
    DepthAnything library, allowing for easy integration and use in projects requiring depth information.

    Attributes:
        image_path (Path): Path to the input image for depth estimation.
        encoder (str): Encoder model to use for depth estimation. Defaults to "vits".
        depth_anything (DepthAnything): The initialized DepthAnything model for depth estimation.
        transform (Compose): A series of transformations applied to the input image before processing.

    Methods:
        process_image(): Estimates the depth of the input image, saves, and returns depth images
                         as numpy arrays
    """

    def __init__(self: DepthAnythingEstimator, image_path: Path, encoder: str = "vits") -> None:
        """Initializes the DepthEstimator with a specific encoder and the path to the input image.

        Parameters:
            image_path (Path): Path to the input image.
            encoder (str): The encoder to use for depth estimation. Defaults to "vits".
        """
        self.image_path = image_path
        self.encoder = encoder
        self.depth_anything = DepthAnything.from_pretrained(
            f"LiheYoung/depth_anything_{encoder}14",
        ).eval()
        self.transform = Compose(
            [
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ],
        )

    def process_image(self: DepthAnythingEstimator) -> tuple[np.ndarray, np.ndarray]:
        """Estimates depth, saves, and returns the depth images as numpy arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of the colored depth image and the raw depth image
                                           as numpy arrays.
        """
        image_filename = self.image_path.stem
        image_np = (
            cv2.cvtColor(cv2.imread(str(self.image_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )

        h, w = image_np.shape[:2]
        transformed_image = self.transform({"image": image_np})["image"]
        image_tensor = torch.from_numpy(transformed_image).unsqueeze(0)

        with torch.no_grad():
            depth = self.depth_anything(image_tensor)

        if len(depth.shape) == 3:
            depth = depth.unsqueeze(0)
        elif len(depth.shape) == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)

        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)

        raw_depth = depth.cpu().numpy().squeeze()
        np.save(output_img / f"{image_filename}_raw_depth.npy", raw_depth)

        depth = F.relu(depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_uint8 = depth.cpu().numpy().astype(np.uint8).squeeze()

        cv2.imwrite(str(output_img / f"{image_filename}_depth_raw.jpg"), depth_uint8)

        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(output_img / f"{image_filename}_depth_colormap.jpg"), depth_colormap)

        return depth_colormap, depth_uint8
