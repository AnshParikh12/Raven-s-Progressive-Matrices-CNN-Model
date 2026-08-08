from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from PIL import Image

from preprocessing import transform
from model import CNNEncoder

data = np.load(
    r".\data\raven_test\distribute_nine\RAVEN_0_train.npz"
)

images = data["image"]

print("Original images:", images.shape)

preprocessed_images = []

for image in images:
    pil_image = Image.fromarray(image)
    tensor = transform(pil_image)
    preprocessed_images.append(tensor)

images_tensor = torch.stack(preprocessed_images)

print("CNN input:", images_tensor.shape)

model = CNNEncoder()

output = model(images_tensor)

print("CNN output:", output.shape)


