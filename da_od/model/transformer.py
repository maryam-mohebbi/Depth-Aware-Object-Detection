import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

# Assuming da_od package provides the necessary imports and configurations
from da_od.config import class_names, output_img, sam_weights, test_img
from da_od.depth_anything.dpt import DepthAnything
from da_od.depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
from da_od.model import seg_sam, yolo_nas
from da_od.segment_anything import SamAutomaticMaskGenerator, SamPredictor

# Depth estimation setup
encoder = "vits"  # Change as needed
depth_anything = DepthAnything.from_pretrained(f"LiheYoung/depth_anything_{encoder}14").eval()
transform = Compose(
    [
        Resize(
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

# Load and prepare the image
image_path = test_img / "img-00001.jpeg"  # Change as per your setup
image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB) / 255.0
image_transformed = transform({"image": image})["image"]
image_transformed = torch.from_numpy(image_transformed).unsqueeze(0)

# Generate depth map
with torch.no_grad():
    depth = depth_anything(image_transformed)

# Process depth map for visualization and use in segmentation
depth = F.interpolate(depth, size=image.shape[:2], mode="bilinear", align_corners=True)
depth = F.relu(depth)  # Ensure non-negative
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.cpu().numpy().astype(np.uint8).squeeze()
depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

# Now, use the depth_colormap as input for segmentation and object detection
class_names_file = class_names / "coco.names.txt"
output_folder_path = output_img / "segmentation.jpg"
checkpoint_path = sam_weights / "sam_vit_h_4b8939.pth"


def read_class_names(file_path):
    with open(file_path) as file:
        return [line.strip() for line in file.readlines()]


classNames = read_class_names(class_names_file)

# The depth_colormap is used as input for the segmentation and detection
# Assuming yolo_nas.get_object_detection() can accept numpy arrays as inputs
bboxes, confidence, labels, class_names, _ = yolo_nas.get_object_detection(depth_colormap)

# Instantiate mask generator
mask_generator = SamAutomaticMaskGenerator(seg_sam.get_model(checkpoint_path))
mask = mask_generator.generate(depth_colormap)
seg_sam.show_anns(mask)

# Create a predictor for SAM with depth_colormap
predictor = SamPredictor(seg_sam.get_model(checkpoint_path))
predictor.set_image(depth_colormap)

combined_mask = np.zeros((depth_colormap.shape[0], depth_colormap.shape[1], 3))
for i, label in enumerate(labels):
    input_box = np.array(bboxes[i])
    mask, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    cls = int(label)
    class_name = classNames[cls]
    x_min, y_min, x_max, y_max = input_box
    cv2.rectangle(depth_colormap, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    cv2.putText(
        depth_colormap,
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

final_image = cv2.addWeighted(depth_colormap, 0.7, combined_mask.astype(np.uint8), 0.3, 0)
plt.figure(figsize=(10, 10))
plt.imshow(final_image)
plt.axis("off")
plt.savefig(str(output_folder_path))
plt.show()
