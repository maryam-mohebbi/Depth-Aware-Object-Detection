from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from da_od.config import model_output, output_img
from da_od.model import DepthDecoder, ResnetEncoder, download_model_if_doesnt_exist


class MonocularDepthEstimator:
    """MonocularDepthEstimator uses a pre-trained model for monocular depth estimation.

    This class is designed to load a specific monocular depth estimation model and use it to estimate
    the depth of an input image. It provides functionalities to process the image, perform depth estimation,
    and return both raw and color-mapped depth images as numpy arrays.

    Attributes:
        image_path (Path): Path to the input image for depth estimation.
        model_name (str): Name of the model to use for depth estimation. Defaults to "mono_640x192".

    Methods:
        load_model(): Loads the encoder and depth decoder models.
        process_image(): Processes the input image, performs depth estimation, and returns the depth images.
    """

    def __init__(self: MonocularDepthEstimator, image_path: Path, model_name: str = "mono_640x192") -> None:
        """Initializes the MonocularDepthEstimator with the path to an input image and the model name.

        Parameters:
            image_path (Path): Path to the input image.
            model_name (str): The model name for depth estimation. Defaults to "mono_640x192".
        """
        self.image_path = image_path
        self.model_name = model_name

        self.models_dir = model_output
        self.encoder_path = self.models_dir / model_name / "encoder.pth"
        self.depth_decoder_path = self.models_dir / model_name / "depth.pth"

        download_model_if_doesnt_exist(model_name)
        self.load_model()

    def load_model(self: MonocularDepthEstimator) -> None:
        """Loads the encoder and depth decoder models from specified paths.

        This method is responsible for loading the ResnetEncoder and DepthDecoder models
        using the model paths defined during initialization. It updates the class attributes
        with the loaded models and their configuration.
        """
        encoder = ResnetEncoder(num_layers=18, pretrained=False)
        depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc.tolist(), scales=list(range(4)))

        loaded_dict_enc = torch.load(self.encoder_path, map_location="cpu")
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(self.depth_decoder_path, map_location="cpu")
        depth_decoder.load_state_dict(loaded_dict)

        encoder.eval()
        depth_decoder.eval()

        self.encoder = encoder
        self.depth_decoder = depth_decoder
        self.feed_height = loaded_dict_enc["height"]
        self.feed_width = loaded_dict_enc["width"]

    def process_image(self: MonocularDepthEstimator) -> tuple[np.ndarray, np.ndarray]:
        """Processes the input image, performs depth estimation, and returns the depth images.

        This method reads the input image, resizes it for the model, performs depth estimation,
        and processes the output to generate both a raw and a color-mapped depth image. It saves
        these images to disk and returns their numpy array representations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the color-mapped depth image and
                                           the normalized raw depth image as numpy arrays.
        """
        input_image = Image.open(self.image_path).convert("RGB")
        original_width, original_height = input_image.size

        input_image_resized = input_image.resize((self.feed_width, self.feed_height), Image.LANCZOS)
        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

        with torch.no_grad():
            features = self.encoder(input_image_pytorch)
            outputs = self.depth_decoder(features)

        disp = outputs["disp_0"]
        disp_resized = torch.nn.functional.interpolate(
            disp,
            (original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )

        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        depth_colormap = plt.get_cmap("magma")(disp_resized_np / vmax)[:, :, :3]
        depth_colormap = (depth_colormap * 255).astype(np.uint8)
        image_filename = self.image_path.stem

        np.save(output_img / f"{image_filename}_raw_depth.npy", disp_resized_np)

        depth_raw_normalized = np.interp(disp_resized_np, (disp_resized_np.min(), vmax), (0, 255)).astype(
            np.uint8,
        )
        cv2.imwrite(str(output_img / f"{image_filename}_depth_raw.jpg"), depth_raw_normalized)

        plt.imsave(str(output_img / f"{image_filename}_depth_colormap.jpg"), depth_colormap)

        return depth_colormap, depth_raw_normalized
