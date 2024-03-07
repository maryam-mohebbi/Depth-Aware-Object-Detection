from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from super_gradients.training import models

from da_od.segment_anything import SamPredictor, sam_model_registry


class PredictionDetail(Protocol):
    """A protocol representing the detailed prediction results from an object detection model.

    Attributes:
        labels (list[str]): A list of label strings for detected objects.
        confidence (list[float]): A list of confidence scores corresponding to each detected object.
        bboxes_xyxy (list[tuple[float, float, float, float]]): A list of bounding boxes for detected objects,
            each represented as a tuple (x1, y1, x2, y2), where (x1, y1) is the top left corner and
            (x2, y2) is the bottom right corner of the bounding box.
    """

    labels: list[str]
    confidence: list[float]
    bboxes_xyxy: list[tuple[float, float, float, float]]


class PredictionResult(Protocol):
    """A protocol for the aggregated result of an object detection prediction.

    This class encapsulates both the class names identified in the detection process
    and the detailed prediction results including labels, confidence scores, and bounding boxes.

    Attributes:
        class_names (list[str]): A list of all class names detected by the model.
        prediction (PredictionDetail): An instance of PredictionDetail containing detailed
            prediction results for each detected object.
    """

    class_names: list[str]
    prediction: PredictionDetail


class ObjectDetector:
    """A class for detecting objects in images using specified models.

    Attributes:
        model (Any): The object detection model.
        conf_threshold (float): The confidence threshold for the detections.

    Args:
        model_name (str): The name of the model to use. Defaults to "yolo_nas_l".
        pretrained_weights (str): The pretrained weights to use with the model. Defaults to "coco".
        conf_threshold (float): The confidence threshold for considering detections. Defaults to 0.25.
    """

    def __init__(
        self: ObjectDetector,
        model_name: str = "yolo_nas_l",
        pretrained_weights: str = "coco",
        conf_threshold: float = 0.25,
    ) -> None:
        """Initializes the ObjectDetector with a specific model,its pretrained weights & confidence threshold.

        This method sets up the object detection model with the specified configurations. It loads the model
        using the provided model name and pretrained weights. The confidence threshold for detections
        is also set, which determines the minimum confidence for detected objects to be considered valid.

        Args:
        model_name (str): The name of the detection model to use. Defaults to "yolo_nas_l", which
            indicates a specific type of YOLO model. The available models are dependent on the
            implementation of the `models.get` method.
        pretrained_weights (str): The name of the dataset on which the model was pretrained. Defaults
            to "coco", indicating the COCO dataset. This parameter is passed directly to the model
            loading mechanism and affects model initialization.
        conf_threshold (float): The minimum confidence score that a detection must have to be
            considered valid. This threshold helps to filter out less confident detections to
            improve the precision of the object detection. Defaults to 0.25.

        The initialization process involves calling an external model loading function, which is expected
        to return an instance of the detection model. This model is then used for object detection tasks.
        """
        self.model = models.get(model_name, pretrained_weights=pretrained_weights)
        self.conf_threshold = conf_threshold

    def get_prediction_info(
        self: ObjectDetector,
        predictions: PredictionResult,
    ) -> tuple[list[tuple[float, float, float, float]], list[float], list[str], list[str]]:
        """Extracts and returns relevant information from prediction results.

        Args:
            predictions: The prediction results from the object detection model.

        Returns:
            A tuple containing the bounding boxes, confidence scores, labels, and class names from
            the predictions.
        """
        class_names = predictions.class_names
        labels = predictions.prediction.labels
        confidence = predictions.prediction.confidence
        bboxes = predictions.prediction.bboxes_xyxy
        return bboxes, confidence, labels, class_names

    def detect_objects(
        self: ObjectDetector,
        image_input: Path | np.ndarray,
    ) -> tuple[list[tuple[float, float, float, float]], list[float], list[str], list[str], np.ndarray]:
        """Detects objects in the given image.

        Args:
            image_input (Path | np.ndarray): The input image, either as a file path or as an
                                             image array (np.ndarray).

        Raises:
            ValueError: If the image input is neither a file path nor an image array.

        Returns:
            tuple[list[Tuple[float, float, float, float]], list[float], list[str], list[str], np.ndarray]:
            The bounding boxes, confidence scores, labels, class names of detected objects, and the
            processed image array.
        """
        if isinstance(image_input, Path):
            image = cv2.imread(str(image_input))
            if image is None:
                path_message = "Failed to load image from the provided file path."
                raise ValueError(path_message)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            formating_path_message = (
                "Image input must be a file path (Path) or an image array (numpy.ndarray)"
            )
            raise TypeError(formating_path_message)

        detection_pred = self.model.predict(image, conf=self.conf_threshold)
        bboxes, confidence, labels, class_names = self.get_prediction_info(detection_pred)
        return bboxes, confidence, labels, class_names, image


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
    """Provides static methods for visualizing different aspects of segmentation and models' outputs.

    This utility class contains methods to display masks, points, bounding boxes, and annotations directly
    on matplotlib Axes objects. It is designed to assist in the visualization of outputs from segmentation
    models, particularly in the context of object detection and segmentation tasks. The visualization tools
    support a variety of visual elements including masks with optional random colors, points with labels,
    bounding boxes, and layered annotations with sorted areas.

    The methods are static, allowing them to be called without the need for instantiating the
    SAMVisualizationTools class. This design choice simplifies quick and on-the-fly visualizations
    during model development and evaluation.

    Methods:
    - show_mask: Plots a segmentation mask on a given Axes object.
    - show_points: Displays positive and negative points with different colors on a given Axes object.
    - show_box: Draws a single bounding box on a given Axes object.
    - show_anns: Overlays multiple segmentation annotations on the current matplotlib figure.
    """

    @staticmethod
    def show_mask(mask: np.ndarray, ax: plt.Axes, *, random_color: bool = False) -> None:
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

        If there are no annotations, the function returns immediately. Otherwise, it sorts the annotations by
        area in descending order and overlays them on the current Axes object.

        Parameters:
        - anns (list[dict]): A list of annotation dictionaries. Each dictionary should have a "segmentation"
                             key with the mask and an "area" key.
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


class SegmentDetection:
    def __init__(self, class_names_file, output_folder_path, checkpoint_path):
        self.class_names_file = class_names_file
        self.output_folder_path = output_folder_path
        self.checkpoint_path = checkpoint_path
        self.class_names = self.read_class_names(self.class_names_file)
        self.model = self.load_model(self.checkpoint_path)
        self.object_detector = ObjectDetector(
            model_name="yolo_nas_l",
            pretrained_weights="coco",
            conf_threshold=0.25,
        )
        self.display_utils = SAMVisualizationTools()

    def read_class_names(self, file_path):
        with open(file_path) as file:
            class_names = [line.strip() for line in file.readlines()]
        return class_names

    def load_model(self, checkpoint_path):
        sam_model_manager = SamModelManager(checkpoint_path)
        model = sam_model_manager.load_model()
        return model

    def detect_and_segment(self, input_data):
        bboxes, _, labels, _, image = self.object_detector.detect_objects(input_data)
        predictor = SamPredictor(self.model)
        predictor.set_image(image)

        combined_mask = np.zeros((image.shape[0], image.shape[1], 3))
        for i, label in enumerate(labels):
            self.process_label(bboxes[i], label, combined_mask, image, predictor)

        self.display_results(image, combined_mask)

    def process_label(self, input_box, label, combined_mask, image, predictor):
        mask, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(input_box)[None, :],
            multimask_output=False,
        )

        cls = int(label)
        class_name = self.class_names[cls]

        x_min, y_min, x_max, y_max = input_box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(
            image,
            class_name,
            (int(x_min), int(y_min) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        mask_color = np.array(random_color)
        combined_mask += mask[0][..., None] * mask_color

    def display_results(self, image, combined_mask):
        final_image = cv2.addWeighted(image, 0.7, combined_mask.astype(np.uint8), 0.3, 0)
        plt.close("all")
        plt.figure(figsize=(10, 10))
        plt.imshow(final_image)
        plt.axis("off")
        plt.savefig(self.output_folder_path)
        plt.show()
