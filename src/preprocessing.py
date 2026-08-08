from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.ToTensor()
])