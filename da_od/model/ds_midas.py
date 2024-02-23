from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from da_od.config import test_img


class MiDaSModel:
    """A class representing the MiDaS model for depth estimation.

    This class encapsulates the functionality of the MiDaS model, allowing for easy
    loading and inference of depth maps from images. It supports different types of
    MiDaS models: 'DPT_Large', 'DPT_Hybrid', and 'MiDaS_small'. 'DPT_Large' offers high accuracy,
    'DPT_Hybrid' provides a balance between performance and resource utilization,
    and 'MiDaS_small' is efficient for limited-resource environments.

    Attributes:
        device (torch.device): The device on which the model will run (CPU or CUDA).
        model (torch.nn.Module): The loaded MiDaS model.
        transforms (torchvision.transforms.Compose): The transformations to apply to input images.
    """

    def __init__(self: MiDaSModel, model_type: str = "DPT_Large") -> None:
        """Initialize the MiDaS depth estimation model.

        Args:
            model_type (str): The type of MiDaS model to be used. Default is 'DPT_Large'.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device)
        self.model.eval()
        self.transforms = Compose(
            [Resize(384), ToTensor(), lambda x: x.to(self.device), lambda x: x.unsqueeze(0)],
        )

    def predict_depth(self: MiDaSModel, image_path: Path) -> torch.Tensor:
        """Predict the depth map of an image given its path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: The estimated depth map as a 2D tensor.
        """
        img = Image.open(image_path)
        input_tensor = self.transforms(img)
        with torch.no_grad():
            depth = self.model(input_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return depth.cpu().numpy()


midas = MiDaSModel()
image_dir = Path(test_img)

fig, axes = plt.subplots(4, 5)
axes = axes.ravel()

for i, img_file in enumerate(os.listdir(image_dir)[:20]):
    img_path = image_dir / img_file
    depth_map = midas.predict_depth(img_path)
    axes[i].imshow(depth_map)
    axes[i].axis("off")
plt.show()
