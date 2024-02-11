import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, SamAutomaticMaskGenerator
import SAM
import YoloNas
import utils

image_path = 'data/img-00001.jpeg'
class_names_file = "data/coco.names.txt"


def main():
    classNames = utils.read_class_names(class_names_file)

    bboxes, confidence, labels, class_names, image = YoloNas.get_object_detection(image_path)

    # Instantiate mask generator
    mask_generator = SamAutomaticMaskGenerator(SAM.sam)
    mask = mask_generator.generate(image)
    SAM.show_anns(mask)

    # Create a predictor for SAM
    image = cv2.imread(image_path)
    predictor = SamPredictor(SAM.sam)
    predictor.set_image(image)

    # Initialize a combined mask
    combined_mask = np.zeros((image.shape[0], image.shape[1], 3))

    # Loop through detections
    for i, label in enumerate(labels):
        input_box = np.array(bboxes[i])

        # Predict mask using SAM
        mask, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False
        )

        # Get the correct class label
        cls = int(label)
        class_name = classNames[cls]

        # Draw bounding box
        x_min, y_min, x_max, y_max = input_box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # Add label
        cv2.putText(image, class_name, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Generate random color for mask
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        mask_color = np.array(random_color)

        # Add the mask to the combined mask with the random color
        combined_mask += mask[0][..., None] * mask_color

    # Overlay the original image with the combined mask
    final_image = cv2.addWeighted(image, 0.7, combined_mask.astype(np.uint8), 0.3, 0)

    # Display the image
    plt.figure(figsize=(18, 18))
    plt.imshow(final_image)
    plt.axis('off')
    plt.savefig('output_folder/prediction.jpg')  # Save the combined output
    plt.show()


if __name__ == "__main__":
    main()
