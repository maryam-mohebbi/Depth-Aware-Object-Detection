import cv2
import numpy as np
import utils
from ultralytics import YOLO

from da_od.config import class_names, yolo_weights

# Constants
yolo_weights_path = yolo_weights / "yolov8n.pt"
class_names_file = class_names / "coco.names.txt"
sample_video_path = "data/samplevideo.mp4"


def main():
    # Open the video file
    cap = cv2.VideoCapture(sample_video_path)
    ## For webcam uncomment these three lines
    # cap = cv2.VideoCapture(0)
    # cap.set(3, 640)
    # cap.set(4, 480)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Load YOLO model
    model = YOLO(yolo_weights_path)
    class_names = utils.read_class_names(class_names_file)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Perform object detection
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Display confidence and class name
                confidence = np.round(box.conf[0], decimals=2)
                cls = int(box.cls[0])
                class_name = class_names[cls]

                text = f"{class_name}: {confidence:.2f}"
                org = (x1, y1 - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (0, 255, 0)
                thickness = 1

                cv2.putText(frame, text, org, font, font_scale, color, thickness)

        # Display the frame
        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    # Release the video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
