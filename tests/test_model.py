from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from model import CNNEncoder


model = CNNEncoder()

images = torch.randn(
    16,
    1,
    80,
    80
)

print("Input:", images.shape)

x = model.conv1(images)
print("After conv1:", x.shape)

x = model.relu(x)
print("After ReLU:", x.shape)

x = model.pool(x)
print("After pool1:", x.shape)

x = model.conv2(x)
print("After conv2:", x.shape)

x = model.relu(x)
print("After ReLU:", x.shape)

x = model.pool(x)
print("After pool2:", x.shape)

x = model.conv3(x)
print("After conv3:", x.shape)

x = model.relu(x)
print("After ReLU:", x.shape)

x = model.pool(x)
print("After pool3:", x.shape)

x = torch.flatten(x, start_dim=1)
print("After flatten:", x.shape)

x = model.fc(x)
print("Final vector:", x.shape)