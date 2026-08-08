from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import RavenDataset

dataset = RavenDataset(r".\data\raven_test\distribute_nine")

print("Number of problems:", len(dataset))

context, choices, target = dataset[0]

print("Context shape:", context.shape)
print("Choices shape:", choices.shape)

print("Target:", target)
print("Target dtype:", target.dtype)