from __future__ import annotations

import hashlib
import logging
import sys
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils import model_zoo
from torchvision import models, transforms

from da_od.config import test_img

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# flake8: noqa: N806, N803, N812


def disp_to_depth(disp: float, min_depth: float, max_depth: float) -> tuple[float, float]:
    """Convert network's sigmoid output into depth prediction.

    This function applies the formula given in the 'additional considerations' section of the paper
    to convert the disparity value to a depth value.

    Args:
    disp (float): The disparity value output by the network.
    min_depth (float): The minimum depth to be considered.
    max_depth (float): The maximum depth to be considered.

    Returns:
    Tuple[float, float]: A tuple containing the scaled disparity and the calculated depth.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(
    axisangle: torch.Tensor,
    translation: torch.Tensor,
    *,
    invert: bool = False,
) -> torch.Tensor:
    """Convert the network's (axisangle, translation) output into a 4x4 transformation matrix.

    This function computes a transformation matrix based on the axis-angle rotation
    and the translation vector provided by the network. It supports optional inversion
    of the transformation.

    Args:
    axisangle (torch.Tensor): The axis-angle rotation tensor.
    translation (torch.Tensor): The translation vector tensor.
    invert (bool, optional): If True, the transformation is inverted. Defaults to False.

    Returns:
    torch.Tensor: The resulting 4x4 transformation matrix.
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    return torch.matmul(R, T) if invert else torch.matmul(T, R)


def get_translation_matrix(translation_vector: torch.Tensor) -> torch.Tensor:
    """Convert a translation vector into a 4x4 transformation matrix.

    This function creates a 4x4 transformation matrix from a given translation vector.
    The matrix is constructed to be used in transformation operations.

    Args:
    translation_vector (torch.Tensor): The translation vector.

    Returns:
    torch.Tensor: The 4x4 transformation matrix constructed from the translation vector.
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)
    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec: torch.Tensor) -> torch.Tensor:
    """Convert an axis-angle rotation into a 4x4 transformation matrix.

    This function takes a batch of axis-angle vectors and converts them into corresponding
    4x4 transformation matrices. The method is adapted from https://github.com/Wallacoloo/printipi.

    Args:
    vec (torch.Tensor): The input tensor of axis-angle vectors, expected shape Bx1x3.

    Returns:
    torch.Tensor: The batch of 4x4 transformation matrices.
    """
    angle = torch.norm(vec, p=2, dim=2, keepdim=True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """A convolutional block that performs a convolution followed by an ELU activation.

    This module applies a 3x3 convolution to the input, followed by an Exponential Linear Unit (ELU)
    activation function.

    Attributes:
    conv (nn.Module): Convolutional layer.
    nonlin (nn.Module): Non-linear activation layer (ELU).
    """

    def __init__(self: ConvBlock, in_channels: int, out_channels: int) -> None:
        """Initializes the ConvBlock module.

        Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        """
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self: ConvBlock, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ConvBlock.

        Args:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying convolution and ELU.
        """
        out = self.conv(x)
        return self.nonlin(out)


class Conv3x3(nn.Module):
    """A convolutional layer that applies padding and then a convolution.

    This module optionally applies reflection padding or zero padding, followed by a 3x3 convolution.

    Attributes:
    pad (nn.Module): Padding layer.
    conv (nn.Conv2d): Convolutional layer.
    """

    def __init__(self: Conv3x3, in_channels: int, out_channels: int, *, use_refl: bool = True) -> None:
        """Initializes the Conv3x3 module.

        Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        use_refl (bool, optional): Flag to use reflection padding if True, otherwise zero padding.
                                   Defaults to True.
        """
        super().__init__()
        self.pad: nn.Module  # Specify the type as nn.Module
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3)

    def forward(self: Conv3x3, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Conv3x3.

        Args:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying padding and convolution.
        """
        out = self.pad(x)
        return self.conv(out)


class BackprojectDepth(nn.Module):
    """A layer to transform a depth image into a point cloud.

    This module takes a depth image and camera intrinsics to back-project pixels into 3D space,
    producing a point cloud.

    Attributes:
    batch_size (int): Batch size.
    height (int): Height of the input image.
    width (int): Width of the input image.
    id_coords (nn.Parameter): Identity coordinates for pixel positions.
    ones (nn.Parameter): Tensor of ones for homogeneous coordinates.
    pix_coords (nn.Parameter): Pixel coordinates in homogeneous form.
    """

    def __init__(self: BackprojectDepth, batch_size: int, height: int, width: int) -> None:
        """Initializes the BackprojectDepth module.

        Args:
        batch_size (int): The size of the batch.
        height (int): The height of the input images.
        width (int): The width of the input images.
        """
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False)

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False,
        )

        self.pix_coords = torch.unsqueeze(
            torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0),
            0,
        )
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self: BackprojectDepth, depth: torch.Tensor, inv_K: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BackprojectDepth.

        Args:
        depth (torch.Tensor): The depth image tensor.
        inv_K (torch.Tensor): The inverted camera intrinsic matrix.

        Returns:
        torch.Tensor: The point cloud obtained from back-projecting the depth image.
        """
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        return torch.cat([cam_points, self.ones], 1)


class Project3D(nn.Module):
    """A layer which projects 3D points into a camera with given intrinsics K and position T.

    This module takes 3D points and projects them onto a 2D plane using the camera intrinsics
    and extrinsics, simulating the perspective projection of a camera.

    Attributes:
    batch_size (int): Batch size.
    height (int): Height of the projected image.
    width (int): Width of the projected image.
    eps (float): A small epsilon value to prevent division by zero.
    """

    def __init__(self: Project3D, batch_size: int, height: int, width: int, eps: float = 1e-7) -> None:
        """Initializes the Project3D module.

        Args:
        batch_size (int): The size of the batch.
        height (int): The height of the output image.
        width (int): The width of the output image.
        eps (float, optional): A small epsilon value for numerical stability. Defaults to 1e-7.
        """
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self: Project3D, points: torch.Tensor, K: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Project3D.

        Args:
        points (torch.Tensor): The 3D points to project.
        K (torch.Tensor): The camera intrinsic matrix.
        T (torch.Tensor): The camera extrinsic matrix.

        Returns:
        torch.Tensor: The projected 2D pixel coordinates.
        """
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        return (pix_coords - 0.5) * 2


def upsample(x: torch.Tensor) -> torch.Tensor:
    """Upsamples an input tensor by a factor of 2.

    This function uses nearest neighbor interpolation to increase the size of the input tensor.

    Args:
    x (torch.Tensor): The input tensor to be upsampled.

    Returns:
    torch.Tensor: The upsampled tensor with dimensions doubled in height and width.
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """Computes the smoothness loss for a disparity image.

    The smoothness loss is edge-aware and uses the color image for calculating gradients.
    The disparity image is expected to be smooth in areas where the color image gradients are low.

    Args:
    disp (torch.Tensor): The disparity image tensor.
    img (torch.Tensor): The color image tensor used for edge-aware smoothness.

    Returns:
    torch.Tensor: The calculated smoothness loss.
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """A layer to compute the Structural Similarity Index (SSIM) loss between a pair of images.

    SSIM is a perceptual metric that quantifies image quality degradation caused by processing such
    as data compression or by losses in data transmission. This class computes the SSIM index between
    two images, which is a measure of the similarity between the two images.

    Attributes:
    mu_x_pool, mu_y_pool (nn.AvgPool2d): Average pooling layers for computing means of x and y.
    sig_x_pool, sig_y_pool, sig_xy_pool (nn.AvgPool2d): Average pooling layers for computing variances and
                                                        covariance.
    refl (nn.ReflectionPad2d): Reflection padding layer.
    C1, C2 (float): Constants for stabilizing divisions.
    """

    def __init__(self: SSIM) -> None:
        """Initializes the SSIM module."""
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self: SSIM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SSIM.

        Args:
        x (torch.Tensor): The first image tensor.
        y (torch.Tensor): The second image tensor.

        Returns:
        torch.Tensor: The SSIM loss between the two images.
        """
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(
    gt: torch.Tensor,
    pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computation of error metrics between predicted and ground truth depths.

    This function calculates various error metrics to evaluate the accuracy of the predicted depth
    against the ground truth. It includes threshold-based accuracy measures, Root Mean Squared Error (RMSE),
    RMSE in log space, absolute relative difference, and squared relative difference.

    Args:
    gt (torch.Tensor): The ground truth depth images.
    pred (torch.Tensor): The predicted depth images.

    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    A tuple containing the calculated error metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3).
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


class DepthDecoder(nn.Module):
    """A depth decoder module for neural networks that decodes encoded features into depth maps.

    This module is designed for use in neural network architectures that require depth estimation,
    such as in depth prediction or 3D reconstruction tasks. It takes the output of an encoder network,
    typically a series of feature maps at different scales, and decodes them into depth maps. The
    decoding process involves a series of up-convolutions and skip connections, optionally utilizing
    features from earlier layers in the encoder for improved detail in the output. The depth maps are
    produced at different scales depending on the specified range.
    """

    def __init__(
        self,
        num_ch_enc: list[int],
        scales: range | int | None = None,
        num_output_channels: int = 1,
        use_skips: bool = True,
    ) -> None:
        """Initializes the DepthDecoder module.

        Args:
            num_ch_enc (List[int]): The number of channels in each layer of the encoder.
            scales (Union[range, List[int]], optional): The scales at which to produce depth maps.
                If not specified, defaults to range(4).
            num_output_channels (int, optional): The number of output channels for the depth maps.
                Defaults to 1.
            use_skips (bool, optional): Whether to use skip connections from the encoder.
                Defaults to True.
        """
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "nearest"
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the DepthDecoder.

        Args:
            input_features (List[torch.Tensor]): A list of tensors representing encoded features from an encoder.

        Returns:
            Dict[str, torch.Tensor]: A dictionary where keys are layer names and values are tensors representing
                                     depth maps at different scales.
        """
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs


class PoseCNN(nn.Module):
    """A convolutional neural network for estimating camera pose from multiple frames.

    This network estimates the relative pose (axis-angle representation of rotation and translation)
    between input frames.

    Attributes:
    num_input_frames (int): Number of input frames.
    convs (dict): Dictionary of convolutional layers.
    pose_conv (nn.Conv2d): Convolutional layer for pose estimation.
    num_convs (int): Number of convolutional layers.
    relu (nn.ReLU): ReLU activation function.
    net (nn.ModuleList): List of convolutional layers as a module list.
    """

    def __init__(self: PoseCNN, num_input_frames: int) -> None:
        """Initializes the PoseCNN module.

        Args:
        num_input_frames (int): Number of frames to input into the network.
        """
        super().__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(inplace=True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self: PoseCNN, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the PoseCNN.

        Args:
        out (torch.Tensor): Input tensor of concatenated frames.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Axisangle and translation components of the estimated pose.
        """
        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


class PoseDecoder(nn.Module):
    """A decoder module for estimating camera pose from encoded features.

    This network takes encoded features from multiple frames and predicts the relative camera pose
    (axis-angle rotation and translation) for each pair of frames.

    Attributes:
    num_ch_enc (List[int]): Number of channels in each layer of the encoder.
    num_input_features (int): Number of input features.
    num_frames_to_predict_for (Optional[int]): Number of frame pairs to predict pose for.
                                               Defaults to num_input_features - 1 if None.
    convs (OrderedDict): Ordered dictionary of convolutional layers.
    relu (nn.ReLU): ReLU activation function.
    net (nn.ModuleList): List of layers as a module list.
    """

    def __init__(
        self: PoseDecoder,
        num_ch_enc: list[int],
        num_input_features: int,
        num_frames_to_predict_for: int | None = None,
        stride: int = 1,
    ) -> None:
        """Initializes the PoseDecoder module.

        Args:
        num_ch_enc (List[int]): Number of channels in each layer of the encoder.
        num_input_features (int): Number of input features.
        num_frames_to_predict_for (Optional[int]): Number of frame pairs to predict pose for.
                                                   Defaults to num_input_features - 1 if None.
        stride (int): Stride for convolutional layers.
        """
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs: OrderedDict[tuple[str, int], nn.Conv2d] = OrderedDict()

        self.convs[("squeeze", 0)] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self: PoseDecoder, input_features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the PoseDecoder.

        Args:
        input_features (List[torch.Tensor]): List of encoded feature tensors from the encoder.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Axisangle and translation components of the estimated pose
                                           for each frame pair.
        """
        last_features = [f[-1] for f in input_features]

        # Create the concatenated tensor from the list of feature tensors
        cat_tensor = torch.cat([self.relu(self.convs[("squeeze", 0)](f)) for f in last_features], 1)

        # Process the concatenated tensor through the pose layers
        out = cat_tensor
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        # Compute the mean and reshape the output
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        # Split the output into axisangle and translation components
        axisangle = out[:, :, :3]
        translation = out[:, :, 3:]

        return axisangle, translation


class ResNetMultiImageInput(models.ResNet):
    """Constructs a ResNet model adapted for multiple input images.

    Attributes:
        inplanes (int): Number of planes in the bottleneck layers.
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): First batch normalization layer.
        relu (nn.ReLU): ReLU activation layer.
        maxpool (nn.MaxPool2d): Max pooling layer.
        layer1 (nn.Sequential): First ResNet layer.
        layer2 (nn.Sequential): Second ResNet layer.
        layer3 (nn.Sequential): Third ResNet layer.
        layer4 (nn.Sequential): Fourth ResNet layer.

    Args:
        block (Type[nn.Module]): Block type for ResNet (BasicBlock or Bottleneck).
        layers (List[int]): Number of layers in each ResNet block.
        num_classes (int): Number of output classes. Default is 1000 for ImageNet.
        num_input_images (int): Number of input images to stack. Default is 1.
    """

    def __init__(
        self: ResNetMultiImageInput,
        block: type[nn.Module],
        layers: list[int],
        num_input_images: int = 1,
    ) -> None:
        """Initializes the ResNetMultiImageInput model.

        This constructor extends the standard ResNet model to handle multiple input images by modifying
        the first convolutional layer to accept a stack of images.

        Args:
            block (Type[nn.Module]): The block type to use within the ResNet model, typically either
                BasicBlock or Bottleneck.
            layers (List[int]): A list containing the number of layers to use in each of the 4 blocks
                of the ResNet model.
            num_input_images (int): The number of input images to be stacked before being fed into the
                network. Defaults to 1. This parameter adjusts the in_channels of the first convolutional
                layer to be `num_input_images * 3`, assuming 3 channels per image (RGB).

        The rest of the network architecture, including batch normalization, ReLU activation, and max
        pooling, follows the standard ResNet configuration. The network also initializes weights using
        kaiming normalization for convolutional layers and constant values for batch normalization layers.
        """
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(
    num_layers: int,
    num_input_images: int = 1,
    *,
    pretrained: bool = False,
) -> nn.Module:
    """Constructs a ResNet model with the option for multiple input images.

    Args:
        num_layers (int): Number of ResNet layers. Must be either 18 or 50.
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        num_input_images (int): Number of frames stacked as input.

    Returns:
        nn.Module: A ResNet model configured with the specified parameters.

    Raises:
        ValueError: If num_layers is not 18 or 50.
    """
    if num_layers not in [18, 50]:
        message = "Invalid num_layers; must be 18 or 50"
        raise ValueError(message)

    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls[f"resnet{num_layers}"])
        loaded["conv1.weight"] = torch.cat([loaded["conv1.weight"]] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """A PyTorch module for a ResNet encoder, capable of handling multiple input images.

    Attributes:
        num_ch_enc (np.array): Array containing the number of channels in each encoder layer.
        encoder (nn.Module): The ResNet encoder model.

    Args:
        num_layers (int): Number of layers in the ResNet model.
        pretrained (bool): If True, use a pre-trained model.
        num_input_images (int): Number of input images. Default is 1.
    """

    def __init__(
        self: ResnetEncoder,
        num_layers: int,
        *,
        pretrained: bool,
        num_input_images: int = 1,
    ) -> None:
        """Initializes the ResnetEncoder module.

        This constructor sets up the encoder based on the number of ResNet layers,
        whether to use a pretrained model, and the number of input images.

        Args:
            num_layers (int): Number of layers in the ResNet model. Valid options include
                              18, 34, 50, 101, and 152.
            pretrained (bool): If True, use a pre-trained model. Otherwise, initialize weights from scratch.
            num_input_images (int): The number of input images to the network. Defaults to 1.
        """
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        if num_layers not in resnets:
            message = f"{num_layers} is not a valid number of resnet layers"
            raise ValueError(message)

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, num_input_images, pretrained=pretrained)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self: ResnetEncoder, input_image: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass of the ResNet encoder.

        Args:
            input_image (torch.Tensor): Input tensor of images.

        Returns:
            List[torch.Tensor]: List of feature maps at different stages of the model.
        """
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


def download_model_if_doesnt_exist(model_name: str) -> None:
    """Downloads and unzips a pre-trained model if it is not already present.

    Args:
        model_name (str): The name of the model to download.

    Raises:
        Exception: If fails to download a file that matches the checksum.
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
            "a964b8356e08a02d009609d9e3928f7c",
        ),
        "stereo_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
            "3dfb76bcff0786e4ec07ac00f658dd07",
        ),
        "mono+stereo_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
            "c024d69012485ed05d7eaa9617a96b81",
        ),
        "mono_no_pt_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
            "9c2f071e35027c895a4728358ffc913a",
        ),
        "stereo_no_pt_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
            "41ec2de112905f85541ac33a854742d1",
        ),
        "mono+stereo_no_pt_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
            "46c3b824f541d143a45c37df65fbab0a",
        ),
        "mono_1024x320": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
            "0ab0766efdfeea89a0d9ea8ba90e1e63",
        ),
        "stereo_1024x320": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
            "afc2f2126d70cf3fdf26b550898b501a",
        ),
        "mono+stereo_1024x320": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
            "cdc5fc9b23513c07d5b19235d9ef08f7",
        ),
    }
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / model_name

    def check_file_matches_md5(checksum: str, fpath: Path) -> bool:
        """Checks if the file at the given path matches the provided MD5 checksum.

        This function reads the entire file into memory to compute its MD5 hash,
        which might not be efficient for very large files.

        Args:
            checksum (str): The expected MD5 checksum.
            fpath (str): The file path to check.

        Returns:
            bool: True if the file's MD5 checksum matches the provided checksum,
                False if it doesn't match or the file doesn't exist.
        """
        path = Path(fpath)
        if not path.exists():
            return False
        with path.open("rb") as f:
            current_sha256checksum = hashlib.sha256(f.read()).hexdigest()
        return current_sha256checksum == checksum

    # see if we have the model already downloaded...
    model_path = Path(model_path)
    encoder_path = model_path / "encoder.pth"
    zip_path = model_path.with_suffix(".zip")

    if not encoder_path.exists():
        model_url, required_md5checksum = download_paths[model_name]
        zip_path = model_path.with_suffix(".zip")

        if not check_file_matches_md5(required_md5checksum, zip_path):
            logger.info("-> Downloading pretrained model to %s", zip_path)
            urllib.request.urlretrieve(model_url, zip_path)  # noqa: S310

        if not check_file_matches_md5(required_md5checksum, zip_path):
            logger.error("Failed to download a file that matches the checksum - quitting")
            sys.exit(1)

        logger.info("   Unzipping model...")
        with zipfile.ZipFile(str(zip_path), "r") as f:
            f.extractall(str(model_path))

        logger.info("Model unzipped to %s", model_path)


model_name = "mono_640x192"
models_dir = Path("models")
encoder_path = models_dir / model_name / "encoder.pth"
depth_decoder_path = models_dir / model_name / "depth.pth"


download_model_if_doesnt_exist(model_name)

# LOADING PRETRAINED MODEL
encoder = ResnetEncoder(num_layers=18, pretrained=False)
depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc.tolist(), scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location="cpu")
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location="cpu")
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval()

image_path = test_img / "img-00002.jpeg"

input_image = Image.open(image_path).convert("RGB")
original_width, original_height = input_image.size

feed_height = loaded_dict_enc["height"]
feed_width = loaded_dict_enc["width"]
input_image_resized = input_image.resize((feed_width, feed_height), Image.LANCZOS)

input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

with torch.no_grad():
    features = encoder(input_image_pytorch)
    outputs = depth_decoder(features)

disp = outputs[("disp", 0)]

disp_resized = torch.nn.functional.interpolate(
    disp,
    (original_height, original_width),
    mode="bilinear",
    align_corners=False,
)

# Saving colormapped depth image
disp_resized_np = disp_resized.squeeze().cpu().numpy()
vmax = np.percentile(disp_resized_np, 95)

plt.subplot(211)
plt.imshow(input_image)
plt.title("Input", fontsize=22)
plt.axis("off")

plt.subplot(212)
plt.imshow(disp_resized_np, cmap="magma", vmax=vmax)
plt.title("Disparity prediction", fontsize=22)
plt.axis("off")
plt.show()
