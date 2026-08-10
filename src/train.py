import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import RavenDataset
from model import CNNEncoder, RavenReasoner

DATA_PATH = r".\data\raven_test\distribute_nine"
CHECKPOINT_PATH = r".\checkpoints\best_model.pth"
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

dataset = RavenDataset(DATA_PATH)

train_size = 80
val_size = 20
generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

cnn = CNNEncoder()
reasoner = RavenReasoner()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(cnn.parameters()) + list(reasoner.parameters()),
    lr=LEARNING_RATE
)

best_val_accuracy = 0.0

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        weights_only=False
    )

    best_val_accuracy = checkpoint["val_accuracy"]

    print(
        f"Previous best validation accuracy: "
        f"{best_val_accuracy:.2%}"
    )

for epoch in range(NUM_EPOCHS):
    cnn.train()
    reasoner.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for context, choices, target in train_loader:
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

        scores = reasoner(
            context_features,
            choice_features
        )

        loss = criterion(scores, target)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = scores.argmax(dim=1)
        
        correct += (predictions == target).sum().item()

        total += target.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total

    cnn.eval()
    reasoner.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for context, choices, target in val_loader:
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

            scores = reasoner(
                context_features,
                choice_features
            )

            loss = criterion(
                scores,
                target
            )

            val_loss += loss.item()

            predictions = scores.argmax(dim=1)

            val_correct += (predictions == target).sum().item()

            val_total += target.size(0)

    val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            {
                "cnn_state_dict": cnn.state_dict(),
                "reasoner_state_dict": reasoner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy
            },
            CHECKPOINT_PATH
        )

        print("Saved new best model!")

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2%} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2%}"
    )