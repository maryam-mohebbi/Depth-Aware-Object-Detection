import cv2
import numpy as np
from super_gradients.training import models


class ObjectDetector:
    def __init__(self, model_name="yolo_nas_l", pretrained_weights="coco", conf_threshold=0.25):
        self.model = models.get(model_name, pretrained_weights=pretrained_weights)
        self.conf_threshold = conf_threshold

    def get_prediction_info(self, predictions):
        class_names = predictions.class_names
        labels = predictions.prediction.labels
        confidence = predictions.prediction.confidence
        bboxes = predictions.prediction.bboxes_xyxy
        return bboxes, confidence, labels, class_names

    def detect_objects(self, image_input):
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Image input must be a file path (string) or an image array (numpy.ndarray)")

        detection_pred = self.model.predict(image, conf=self.conf_threshold)
        bboxes, confidence, labels, class_names = self.get_prediction_info(detection_pred)
        return bboxes, confidence, labels, class_names, image
