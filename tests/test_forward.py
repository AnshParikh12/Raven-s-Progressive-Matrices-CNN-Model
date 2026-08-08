from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RavenDataset
from model import CNNEncoder, RavenReasoner

dataset = RavenDataset(
    r".\data\raven_test\distribute_nine"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

cnn = CNNEncoder()
reasoner = RavenReasoner()
optimizer = torch.optim.Adam(
    list(cnn.parameters()) + list(reasoner.parameters()),
    lr=0.001
)

context, choices, target = next(iter(loader))

print("Context:", context.shape)
print("Choices:", choices.shape)
print("Target:", target.shape)

batch_size = context.shape[0]

context = context.view(
    batch_size * 8,
    1,
    80,
    80
)

choices = choices.view(
    batch_size * 8,
    1,
    80,
    80
)

context_features = cnn(context)
choice_features = cnn(choices)

print("Context features:", context_features.shape)
print("Choice features:", choice_features.shape)

context_features = context_features.view(
    batch_size,
    8,
    256
)

choice_features = choice_features.view(
    batch_size,
    8,
    256
)

print("Context features reshaped:", context_features.shape)
print("Choice features reshaped:", choice_features.shape)

optimizer.zero_grad()

scores = reasoner(
    context_features,
    choice_features
)

criterion = nn.CrossEntropyLoss()

loss = criterion(scores, target)

loss.backward()

optimizer.step()

print("CNN conv1 gradient:")
print(cnn.conv1.weight.grad.shape)

print("CNN conv1 gradient mean:")
print(cnn.conv1.weight.grad.mean())

print("Reasoner first layer gradient:")
print(reasoner.scorer[0].weight.grad.shape)

print("Reasoner first layer gradient mean:")
print(reasoner.scorer[0].weight.grad.mean())

print("Scores:")
print(scores)

print("Target:")
print(target)

print("Loss before update:")
print(loss.item())

# Second forward pass after weights were updated

context_features = cnn(context)
choice_features = cnn(choices)

context_features = context_features.view(
    batch_size,
    8,
    256
)

choice_features = choice_features.view(
    batch_size,
    8,
    256
)

new_scores = reasoner(
    context_features,
    choice_features
)

new_loss = criterion(new_scores, target)

print("Loss after update:")
print(new_loss.item())

print("New scores:")
print(new_scores)