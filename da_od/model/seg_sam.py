from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from da_od.segment_anything import sam_model_registry


class SamModelManager:
    """Manages the loading and handling of SAM (Segment Anything Model) models.

    This manager facilitates the initialization of models from a checkpoint file and
    ensures they are loaded onto the appropriate device (CPU or GPU, depending on availability).
    The type of SAM model to be loaded can be specified, with support for different architectures
    defined in the `sam_model_registry`.
    """

    def __init__(self: SamModelManager, checkpoint_path: str, model_type: str = "vit_h") -> None:
        """Initializes the SamModelManager with a given checkpoint path and model type.

        Parameters:
        - checkpoint_path (str): The file path to the model checkpoint.
        - model_type (str, optional): The type of model to load. Defaults to "vit_h".

        Attributes:
        - checkpoint_path (str): Path to the model checkpoint.
        - model_type (str): Type of the model.
        - device (torch.device): Device on which the model will be loaded (GPU or CPU).
        - model (Any): Loaded model based on the model type and checkpoint.
        """
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

    def load_model(self: SamModelManager) -> torch.nn.Module:
        """Loads a SAM model from the specified checkpoint into the designated device.

        The model type is selected based on the `model_type` attribute, and the actual model
        instance is retrieved from the `sam_model_registry`.

        Returns:
        - The loaded SAM model as an instance of `torch.nn.Module`.
        """
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(device=self.device)
        return sam


class SAMVisualizationTools:
    @staticmethod
    def show_mask(mask: np.ndarray, ax: plt.Axes, random_color: bool = False) -> None:
        """Displays a mask overlay on the given Axes object with an optional random color.

        Parameters:
        - mask (np.ndarray): The mask to display.
        - ax (plt.Axes): The matplotlib Axes object to display the mask on.
        - random_color (bool, optional): Whether to use a random color for the mask. Defaults to False.
        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords: np.ndarray, labels: np.ndarray, ax: plt.Axes, marker_size: int = 365) -> None:
        """Displays points on the given Axes object, colored based on their labels.

        Parameters:
        - coords (np.ndarray): Coordinates of the points to display.
        - labels (np.ndarray): Labels for each point, used to determine the color.
        - ax (plt.Axes): The matplotlib Axes object to display the points on.
        - marker_size (int, optional): Size of the markers. Defaults to 365.
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )

    @staticmethod
    def show_box(box: tuple[float, float, float, float], ax: plt.Axes) -> None:
        """Draws a box on the given Axes object.

        Parameters:
        - box (tuple[float, float, float, float]): The coordinates of the box as (x_min, y_min, x_max, y_max).
        - ax (plt.Axes): The matplotlib Axes object to draw the box on.
        """
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))

    @staticmethod
    def show_anns(anns: list[dict]) -> None:
        """Displays annotations on the current matplotlib figure.

        If there are no annotations, the function returns immediately. Otherwise, it sorts the annotations by area in descending order and overlays them on the current Axes object.

        Parameters:
        - anns (list[dict]): A list of annotation dictionaries. Each dictionary should have a "segmentation" key with the mask and an "area" key.
        """
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann["segmentation"]
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
