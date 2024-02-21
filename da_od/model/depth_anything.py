import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms import Compose

from da_od.config import output_img, test_img
from da_od.depth_anything.dpt import DepthAnything
from da_od.depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

encoder = "vits"  # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained(
    f"LiheYoung/depth_anything_{encoder}14",
).eval()

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

image_path = test_img / "img-00015.jpg"
image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB) / 255.0
h, w = image.shape[:2]
image = transform({"image": image})["image"]
image = torch.from_numpy(image).unsqueeze(0)

# depth shape: 1xHxW
with torch.no_grad():
    depth = depth_anything(image)


# Ensure depth has the correct shape (N, C, H, W) before interpolation
if len(depth.shape) == 3:
    depth = depth.unsqueeze(0)  # Add a batch dimension if needed
elif len(depth.shape) == 2:
    depth = depth.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions if needed

# Resizing to original image size
depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)

# Saving the raw depth data
raw_depth = depth.cpu().numpy().squeeze()
np.save(output_img / "raw_depth.npy", raw_depth)

# Apply ReLU to ensure all values are non-negative
depth = F.relu(depth)

# Normalize to 0-255 for visualization
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.cpu().numpy().astype(np.uint8).squeeze()

# Apply color map for visualization
depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

# Save or display the result
cv2.imwrite(str(output_img / "depth_colormap.png"), depth_colormap)
cv2.imshow("Depth Colormap", depth_colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()
