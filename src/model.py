import torch
import torch.nn as nn
import torchvision
from PIL import Image
import numpy as np

class CNNEncoder(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )


        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.fc = nn.Linear(
            128*10*10,
            embedding_dim
        )

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)

        return x

class RavenReasoner(nn.Module):

    def __init__(self):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(256*9, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, context_features, choice_features):
        batch_size = context_features.shape[0]

        scores = []

        for i in range(8):
            candidate = choice_features[:, i, :]

            combined = torch.cat(
                [context_features, candidate.unsqueeze(1)],
                dim=1
            )

            combined = combined.reshape((batch_size, 9 * 256))

            score = self.scorer(combined)

            scores.append(score)

        scores = torch.cat(
            scores, dim=1
        )

        return scores



