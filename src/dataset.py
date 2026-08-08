import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocessing import transform

class RavenDataset(Dataset):

    def __init__(self, data_dir):
        self.files = glob.glob(
            f"{data_dir}/*.npz"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]

        data = np.load(file_path)

        images = data["image"]
        target = data["target"]

        processed_images = []

        for image in images:
            pil_image = Image.fromarray(image)
            tensor = transform(pil_image)
            processed_images.append(tensor)

        images = torch.stack(processed_images)

        context = images[:8]
        choices = images[8:]

        target = torch.tensor(
            target,
            dtype=torch.long
        )

        return context, choices, target
