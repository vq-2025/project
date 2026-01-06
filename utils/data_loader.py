import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
from pathlib import Path


class ImageFolderDataset(Dataset):
    """Custom dataset for loading images from a folder"""
    def __init__(self, root_dir, image_size=256, transform=None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        # Find all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.root_dir.rglob(ext)))

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return dummy label for compatibility


def get_data_loaders(
    dataset_name='cifar10',
    data_dir='./data',
    batch_size=32,
    image_size=256,
    num_workers=4,
    pin_memory=True
):
    """Get train and validation data loaders"""

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if dataset_name.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        val_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=val_transform
        )

    elif dataset_name.lower() == 'imagenet':
        train_dataset = datasets.ImageNet(
            root=data_dir,
            split='train',
            transform=transform
        )
        val_dataset = datasets.ImageNet(
            root=data_dir,
            split='val',
            transform=val_transform
        )

    elif dataset_name.lower() == 'celeba':
        train_dataset = datasets.CelebA(
            root=data_dir,
            split='train',
            download=True,
            transform=transform
        )
        val_dataset = datasets.CelebA(
            root=data_dir,
            split='valid',
            download=True,
            transform=val_transform
        )

    elif dataset_name.lower() == 'custom':
        train_dataset = ImageFolderDataset(
            os.path.join(data_dir, 'train'),
            image_size=image_size,
            transform=transform
        )
        val_dataset = ImageFolderDataset(
            os.path.join(data_dir, 'val'),
            image_size=image_size,
            transform=val_transform
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader


class AugmentedDataset(Dataset):
    """Dataset with advanced augmentations"""
    def __init__(self, base_dataset, image_size=256):
        self.base_dataset = base_dataset
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        if isinstance(img, torch.Tensor):
            # Convert back to PIL for augmentation
            img = transforms.ToPILImage()(img)

        img = self.transform(img)
        return img, label
