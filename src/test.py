import torch
from torch.utils.data import DataLoader, random_split

from dataset import RavenDataset
from model import CNNEncoder, RavenReasoner

DATA_PATH = r".\data\raven_test\distribute_nine"
CHECKPOINT_PATH = r".\checkpoints\best_model.pth"
BATCH_SIZE = 4

dataset = RavenDataset(DATA_PATH)

train_size = 80
val_size = 20
generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

cnn = CNNEncoder()
reasoner = RavenReasoner()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    weights_only=False
)

cnn.load_state_dict(
    checkpoint["cnn_state_dict"]
)

reasoner.load_state_dict(
    checkpoint["reasoner_state_dict"]
)

print(
    f"Loaded checkpoint from epoch "
    f"{checkpoint['epoch']}"
)

print(
    f"Checkpoint validation accuracy: "
    f"{checkpoint['val_accuracy']:.2%}"
)

cnn.eval()
reasoner.eval()

correct = 0
total = 0

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

        predictions = scores.argmax(dim=1)

        correct+=(predictions == target).sum().item()

        total+=target.size(0)

accuracy = correct / total

print(
    f"Validation Accuracy: "
    f"{accuracy:.2%}"
)