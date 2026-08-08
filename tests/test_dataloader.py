from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader

from dataset import RavenDataset

dataset = RavenDataset(r".\data\raven_test\distribute_nine")

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


context, choices, target = next(iter(loader))


print("Context shape:", context.shape)
print("Choices shape:", choices.shape)
print("Target shape:", target.shape)

print("Targets:", target)