from typing import Union

import cv2
import numpy as np
from super_gradients.training import models


def get_prediction_info(predictions):
    class_names = predictions.class_names
    labels = predictions.prediction.labels
    confidence = predictions.prediction.confidence
    bboxes = predictions.prediction.bboxes_xyxy
    return bboxes, confidence, labels, class_names


def get_object_detection(image_input: Union[str, np.ndarray]):
    # Check if the input is a string (path) or an image array
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Image input must be a file path (string) or an image array (numpy.ndarray)")

    model = models.get("yolo_nas_l", pretrained_weights="coco")

    conf_threshold = 0.25

    # If you have the image as a NumPy array, adjust the predict method to handle it directly
    # This assumes model.predict can work with NumPy arrays; if not, the model or predict method needs to be adapted
    if isinstance(image_input, np.ndarray):
        detection_pred = model.predict(image, conf=conf_threshold)  # Adjusted to use the image array
    else:
        detection_pred = model.predict(str(image_input), conf=conf_threshold)

    bboxes, confidence, labels, class_names = get_prediction_info(detection_pred)

    return bboxes, confidence, labels, class_names, image
