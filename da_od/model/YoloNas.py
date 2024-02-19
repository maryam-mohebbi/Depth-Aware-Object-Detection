import cv2
from super_gradients.training import models

def get_prediction_info(predictions):
    class_names = predictions.class_names
    labels = predictions.prediction.labels
    confidence = predictions.prediction.confidence
    bboxes = predictions.prediction.bboxes_xyxy
    return bboxes, confidence, labels, class_names


def get_object_detection(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = models.get('yolo_nas_l', pretrained_weights="coco")

    conf_threshold = 0.25
    detection_pred = model.predict(str(image_path), conf=conf_threshold)
    bboxes, confidence, labels, class_names = get_prediction_info(detection_pred)

    return bboxes, confidence, labels, class_names, image
