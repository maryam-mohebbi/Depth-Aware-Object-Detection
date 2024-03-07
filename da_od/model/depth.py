from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor

from da_od.config import model_output, output_img
from da_od.model.depth_anything.dpt import DepthAnything
from da_od.model.depth_anything.util.transform import NormalizeImage, PrepareForNet
from da_od.model.depth_anything.util.transform import Resize as DA_Resize
from da_od.model.monodepth import DepthDecoder, ResnetEncoder, download_model_if_doesnt_exist


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
                DA_Resize(
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
        np.save(output_img / f"DA_{image_filename}_raw_depth.npy", raw_depth)

        depth = F.relu(depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_uint8 = depth.cpu().numpy().astype(np.uint8).squeeze()

        cv2.imwrite(str(output_img / f"DA_{image_filename}_depth_raw.jpg"), depth_uint8)

        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(output_img / f"DA_{image_filename}_depth_colormap.jpg"), depth_colormap)

        return depth_colormap, depth_uint8


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

        depth_min = depth_numpy.min()
        depth_max = depth_numpy.max()
        depth_normalized = (depth_numpy - depth_min) / (depth_max - depth_min) * 255.0
        depth_uint8 = depth_normalized.astype(np.uint8)

        raw_depth_image_path = output_img / f"Mi_{self.image_path.stem}_depth_raw.jpg"
        cv2.imwrite(str(raw_depth_image_path), depth_uint8)

        raw_depth_data_path = output_img / f"Mi_{self.image_path.stem}_raw_depth.npy"
        np.save(raw_depth_data_path, depth_numpy)

        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        colored_depth_image_path = output_img / f"Mi_{self.image_path.stem}_depth_colormap.jpg"
        cv2.imwrite(str(colored_depth_image_path), depth_colored)

        return depth_colored, depth_uint8


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

        np.save(output_img / f"Mo_{image_filename}_raw_depth.npy", disp_resized_np)

        depth_raw_normalized = np.interp(disp_resized_np, (disp_resized_np.min(), vmax), (0, 255)).astype(
            np.uint8,
        )
        cv2.imwrite(str(output_img / f"Mo_{image_filename}_depth_raw.jpg"), depth_raw_normalized)

        plt.imsave(str(output_img / f"Mo_{image_filename}_depth_colormap.jpg"), depth_colormap)

        return depth_colormap, depth_raw_normalized
