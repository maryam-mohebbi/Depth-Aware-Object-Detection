from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from da_od.config import output_img, test_img
from da_od.depth_anything.dpt import DepthAnything
from da_od.depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize


class DepthEstimator:
    def __init__(self, image_path, encoder="vits"):
        self.image_path = image_path
        self.encoder = encoder
        self.depth_anything = DepthAnything.from_pretrained(
            f"LiheYoung/depth_anything_{encoder}14",
        ).eval()
        self.transform = Compose(
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

    def process_image(self):
        image_filename = Path(self.image_path).stem
        image = cv2.cvtColor(cv2.imread(str(self.image_path)), cv2.COLOR_BGR2RGB) / 255.0
        h, w = image.shape[:2]
        image = self.transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0)

        with torch.no_grad():
            depth = self.depth_anything(image)

        if len(depth.shape) == 3:
            depth = depth.unsqueeze(0)
        elif len(depth.shape) == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)

        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)

        raw_depth = depth.cpu().numpy().squeeze()
        raw_depth_path = output_img / f"{image_filename}_raw_depth.npy"
        np.save(raw_depth_path, raw_depth)

        depth = F.relu(depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_uint8 = depth.cpu().numpy().astype(np.uint8).squeeze()

        cv2.imwrite(str(output_img / f"{image_filename}_depth_raw.jpg"), depth_uint8)

        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(output_img / f"{image_filename}_depth_colormap.jpg"), depth_colormap)

        return depth_colormap, depth_uint8


image_path = test_img / "img-00019.jpg"
depth_estimator = DepthEstimator(image_path, encoder="vits")
depth_img_colored, depth_raw = depth_estimator.process_image()
