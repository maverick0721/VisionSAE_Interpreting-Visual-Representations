import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloader(cfg):
    transform = transforms.Compose([
        transforms.Resize(cfg["dataset"]["image_size"]),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return loader