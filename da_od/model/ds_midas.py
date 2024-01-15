import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from da_od.config import test_img

# Load the MiDaS model
model_type = "DPT_Large"  # Options: DPT_Large, DPT_Hybrid, MiDaS_small
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Define the standard MiDaS transforms
midas_transforms = Compose(
    [
        Resize(384),
        ToTensor(),
        lambda x: x.to(device),
        lambda x: x.unsqueeze(0),
    ],
)

# Load an image (replace with your own image path)
img = Image.open(test_img / "street01.jpg")

# Apply transforms
input_tensor = midas_transforms(img)

# Predict depth
with torch.no_grad():
    depth = midas(input_tensor)

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=img.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth = depth.cpu().numpy()

# Display depth map
plt.imshow(depth)
plt.axis("off")
plt.show()
