from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from super_gradients.training import models


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
