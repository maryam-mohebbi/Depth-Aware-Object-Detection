from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from da_od.config import model_output, test_img
from da_od.model import DepthDecoder, ResnetEncoder, download_model_if_doesnt_exist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


model_name = "mono_640x192"
models_dir = model_output
encoder_path = models_dir / model_name / "encoder.pth"
depth_decoder_path = models_dir / model_name / "depth.pth"


download_model_if_doesnt_exist(model_name)

# LOADING PRETRAINED MODEL
encoder = ResnetEncoder(num_layers=18, pretrained=False)
depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc.tolist(), scales=list(range(4)))

loaded_dict_enc = torch.load(encoder_path, map_location="cpu")
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location="cpu")
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval()

image_path = test_img / "img-00002.jpeg"

input_image = Image.open(image_path).convert("RGB")
original_width, original_height = input_image.size

feed_height = loaded_dict_enc["height"]
feed_width = loaded_dict_enc["width"]
input_image_resized = input_image.resize((feed_width, feed_height), Image.LANCZOS)

input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

with torch.no_grad():
    features = encoder(input_image_pytorch)
    outputs = depth_decoder(features)

disp = outputs["disp_0"]


disp_resized = torch.nn.functional.interpolate(
    disp,
    (original_height, original_width),
    mode="bilinear",
    align_corners=False,
)

# Saving colormapped depth image
disp_resized_np = disp_resized.squeeze().cpu().numpy()
vmax = np.percentile(disp_resized_np, 95)

plt.subplot(211)
plt.imshow(input_image)
plt.title("Input", fontsize=22)
plt.axis("off")

plt.subplot(212)
plt.imshow(disp_resized_np, cmap="magma", vmax=vmax)
plt.title("Disparity prediction", fontsize=22)
plt.axis("off")
plt.show()
