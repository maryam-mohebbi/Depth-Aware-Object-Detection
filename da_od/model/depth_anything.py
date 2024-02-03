import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms import Compose

from da_od.config import test_img
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

image_path = test_img / "img-00001.jpeg"
image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB) / 255.0
h, w = image.shape[:2]
image = transform({"image": image})["image"]
image = torch.from_numpy(image).unsqueeze(0)

# depth shape: 1xHxW
with torch.no_grad():
    depth = depth_anything(image)

depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

depth = depth.cpu().numpy().astype(np.uint8)
depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

cv2.imshow("", depth)

cv2.waitKey(0)
