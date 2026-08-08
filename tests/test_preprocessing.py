from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from preprocessing import transform


data = np.load(
    r".\data\raven_test\distribute_nine\RAVEN_0_train.npz"
)

images = data["image"]

print("Original shape:", images.shape)
print("Original dtype:", images.dtype)


image = images[0]

pil_image = Image.fromarray(image)

tensor = transform(pil_image)

print("Tensor shape:", tensor.shape)
print("Tensor dtype:", tensor.dtype)
print("Minimum value:", tensor.min())
print("Maximum value:", tensor.max())